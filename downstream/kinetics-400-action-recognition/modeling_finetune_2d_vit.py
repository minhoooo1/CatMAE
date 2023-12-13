# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        self.patch_embed = torch.nn.Conv3d(self.patch_embed.proj.in_channels, self.patch_embed.proj.out_channels, 
                                    kernel_size=(2, 16, 16),
                                    stride=(2, 16, 16),
                                    padding=(0, 0, 0))
        self.patch_embed.patch_size = (16, 16)

    def get_num_layers(self):
        return len(self.blocks)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embed[:,:1]
        x = x + self.pos_embed[:,1:].repeat(1, 8, 1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        
        return outcome


def vit_small_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == "__main__":
    model = vit_base_patch16_224()
    x = torch.randn(1,3,16,224,224)
    y = model(x)
    print(y.shape)
    # model_sd = model.state_dict()

    # # pretrain_sd = torch.load("/home/jiangzhouqiang/projects/VideoMAE/checkpoint/mae_pretrain_vit_base.pth")["model"]
    # # # pretrain_sd["x"] = 1
    # # # pretrain_sd["xx"] = 2
    # # print( len(model_sd.keys()), len(pretrain_sd.keys()))
    # for i,j in model_sd.items():
    #     print(f"{i}\t{j.shape}")
    # # for i in pretrain_sd.keys():
    #     if i not in model_sd:
    #         print(i)