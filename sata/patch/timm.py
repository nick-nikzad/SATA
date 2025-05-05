# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from typing import Tuple
import math
import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from sata.sata import SATA

class SATA_Block(Block):
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_attn, M_att_h = self.attn(self.norm1(x))
        x = x + self._drop_path1(x_attn)

        
        residual_tokens = None
        sata_chk = self._sata_info["sata_block"].pop(0)
        
        if sata_chk==1: ## it is equivalent to layer_number >= gamma*number of blocks 
            x, residual_tokens = SATA(x,M_att=M_att_h.mean(1))        
        
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        
        if residual_tokens is not None:
            x = torch.cat((x,residual_tokens),dim=1)
        return x


class SATA_Attention(Attention):
    """
    Modifications:
     - Return the attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1))* self.scale


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return attn as well here
        return x, attn

def parse_sata_block(num_blocks, gamma=0.7):
    num_ones = math.ceil((1-gamma) * num_blocks)
    result = [0] * (num_blocks - num_ones) + [1] * num_ones
    return result

def make_sata_class(transformer_class):
    class SATA_VisionTransformer(transformer_class):

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._sata_info["sata_block"] = parse_sata_block(len(self.blocks), gamma=self.gamma)

            return super().forward(*args, **kwdargs)

    return SATA_VisionTransformer


def apply_patch(
    model: VisionTransformer,
    gamma:float = 0.7,
    alpha:float = 1.0):
    """
    Applies SATA to this transformer. 

    """
    SATA_VisionTransformer = make_sata_class(model.__class__)

    model.__class__ = SATA_VisionTransformer
    model.alpha = alpha
    model.gamma = gamma
    
    model._sata_info = {
        "alpha": model.alpha,
        "gamma": model.gamma,
        "sata_block":[]
    }

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = SATA_Block
            module._sata_info = model._sata_info
        elif isinstance(module, Attention):
            module.__class__ = SATA_Attention
