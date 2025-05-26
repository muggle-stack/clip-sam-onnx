#!/usr/bin/env python3
"""
Text-guided SAM segmentation – robust version
============================================
* Uses **CLIPSeg** heat-map to generate a *bounding box + foreground point* and feeds
  them to **SAM** (Segment-Anything) for high-quality refinement.
* Fallbacks gracefully if CLIPSeg finds no region (optionally passes a coarse mask).

Quick start
-----------
```bash
python inference_clip_sam.py \
       --img images/dog.jpg \
       --text "a dog" \
       --clipseg-model models/clipseg-rd16_fp16.onnx \
       --vit-model models/sam_vit_b_encoder.onnx \
       --decoder-model models/sam_decoder.onnx \
       --device cuda --output dog_out.jpg
```
Additional flags:
* `--thr 0.28`  Probability threshold for CLIPSeg (default 0.28)
* `--kernel 15`  Dilation size (pixel) before connected-components (default 15)

Dependencies: `opencv-python`, `onnxruntime`, `transformers` (for tokenizer).
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from sam.predictor import SamPredictor

###############################################################################
# CLIPSeg wrapper                                                              #
###############################################################################
class CLIPSeg:
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
    STD = np.array([0.26862954, 0.26130258, 0.27577711])

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        so = ort.SessionOptions()
        prov = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, so, providers=prov)
        self._build_tokenizer()

    def _build_tokenizer(self):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # ---------------------------------------------------------------------
    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (352, 352), interpolation=cv2.INTER_LINEAR)
        rgb = (rgb - CLIPSeg.MEAN) / CLIPSeg.STD
        return rgb.transpose(2, 0, 1)[None]

    def _tokenize(self, text: str):
        t = self.tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="np")
        return t["input_ids"].astype(np.int64), t["attention_mask"].astype(np.int64)

    # ---------------------------------------------------------------------
    def __call__(self, img_bgr: np.ndarray, prompt: str) -> np.ndarray:
        pix = self._preprocess(img_bgr).astype(np.float32)
        ids, attn = self._tokenize(prompt)
        logits = self.session.run(None, {
            "pixel_values": pix,
            "input_ids": ids,
            "attention_mask": attn,
        })[0]
        if logits.ndim == 4:
            logits = logits[0, 0]
        elif logits.ndim == 3:
            logits = logits[0]
        elif logits.ndim != 2:
            raise RuntimeError(f"Unexpected CLIPSeg output shape {logits.shape}")
        prob = 1 / (1 + np.exp(-logits.astype(np.float32)))  # sigmoid
        return prob  # (352,352)

###############################################################################
# Utility functions                                                            #
###############################################################################

def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.6, color=(0, 0, 255)) -> np.ndarray:
    """Overlay binary mask on BGR image (mask assumed 0/1)."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    color_img = np.zeros_like(img)
    color_img[:] = color
    masked = cv2.bitwise_and(color_img, color_img, mask=mask_u8)
    return cv2.addWeighted(img, 1 - alpha, masked, alpha, 0)

###############################################################################
# CLI                                                                         #
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--img", required=True, help="Input image path")
    p.add_argument("--vit-model", required=True, help="SAM ViT encoder")
    p.add_argument("--decoder-model", required=True, help="SAM prompt+decoder ONNX")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--warmup", type=int, default=0, help="Warm-up iterations for SAM")
    # text mode
    p.add_argument("--clipseg-model", default="", help="CLIPSeg ONNX model path")
    p.add_argument("--text", default="", help="Text prompt for target")
    p.add_argument("--thr", type=float, default=0.28, help="Probability threshold for CLIPSeg")
    p.add_argument("--kernel", type=int, default=15, help="Dilation kernel size before CCA")
    # output
    p.add_argument("--output", default="", help="Save overlay to path (optional)")
    return p.parse_args()

###############################################################################
# Main                                                                        #
###############################################################################

def largest_component(mask_bin: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    num, labels, stats, cent = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return mask_bin, [], []
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    sel = (labels == idx).astype(np.uint8)
    x, y, w, h, _ = stats[idx]
    cx, cy = cent[idx]
    return sel, [x, y, x + w, y + h], [int(cx), int(cy)]


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)

    predictor = SamPredictor(args.vit_model, args.decoder_model, args.device, args.warmup)
    predictor.register_image(img)

    # ------------------------------------------------------------------
    if args.text and args.clipseg_model:
        clipseg = CLIPSeg(args.clipseg_model, args.device)
        prob = clipseg(img, args.text)                 # (352,352)
        prob = cv2.resize(prob, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)

        mask_bin = (prob > args.thr).astype(np.uint8)
        if args.kernel > 1:
            kernel = np.ones((args.kernel, args.kernel), np.uint8)
            mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)

        sel, box, point = largest_component(mask_bin)
        if not box:
            raise RuntimeError("CLIPSeg failed to find a region – try lowering --thr or using a better model")

        sam_out = predictor.get_mask(point_coords=[point], point_labels=[1], boxes=[box])
        mask = sam_out["masks"][0, 0]
        full_mask = mask  # already image size because predictor knows original size

        # ---------------- save & show ----------------
        vis = overlay(img, full_mask)
        if args.output:
            cv2.imwrite(args.output, vis)
            mask_path = Path(args.output).with_name(Path(args.output).stem + "_mask.png")
            cv2.imwrite(str(mask_path), (full_mask > 0).astype(np.uint8) * 255)
        cv2.imshow("SAM text", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ------------------------------------------------------------------ interactive fallback
    points: List[List[int]] = []
    boxes: List[List[int]] = []
    tmp: List[int] = []
    first = True

    def redraw(base=None):
        canvas = img.copy() if base is None else base.copy()
        for bx in boxes:
            cv2.rectangle(canvas, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
        for px, py in points:
            cv2.circle(canvas, (px, py), 5, (255, 0, 0), -1)
        cv2.imshow("SAM", canvas)
        return canvas

    def mouse(event, x, y, *_):
        nonlocal first, tmp, last
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            m = predictor.get_mask(points, [1]*len(points), boxes if boxes else None)["masks"][0, 0]
            last = redraw(overlay(img, cv2.resize(m, img.shape[1::-1], cv2.INTER_NEAREST)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if first:
                tmp = [x, y]; first = False
            else:
                tmp += [x, y]
                x1, y1, x2, y2 = tmp
                boxes.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
                first, tmp = True, []
                m = predictor.get_mask(points if points else None, [1]*len(points) if points else None, boxes)["masks"][0, 0]
                last = redraw(overlay(img, cv2.resize(m, img.shape[1::-1], cv2.INTER_NEAREST)))

    last = img.copy()
    cv2.namedWindow("SAM")
    cv2.setMouseCallback("SAM", mouse)
    redraw()
    key = cv2.waitKey(0)
    if (key & 0xFF) == ord("s") and args.output:
        cv2.imwrite(args.output, last)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
