# export_encoder.py
import torch
from functools import partial
from collections import OrderedDict
from segment_anything.modeling.image_encoder import ImageEncoderViT

ckpt = torch.load("sam_vit_b_01ec64.pth", map_location="cpu")

# 建同规格 vit_b encoder
encoder = ImageEncoderViT(
    depth=12, embed_dim=768, img_size=1024,
    mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    num_heads=12, patch_size=16, qkv_bias=True,
    use_rel_pos=True, global_attn_indexes=[2,5,8,11],
    window_size=14, out_chans=256)

# 只加载 image_encoder.* 权重
state = OrderedDict({k[14:]: v for k,v in ckpt.items() if k.startswith("image_encoder.")})
encoder.load_state_dict(state, strict=True)
encoder.eval()

dummy = torch.randn(1, 3, 1024, 1024)
torch.onnx.export(encoder, dummy, "./sam_vit_b_encoder.onnx",
                  opset_version=17, input_names=["input"], output_names=["output"])

