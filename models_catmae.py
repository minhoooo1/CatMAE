# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed



class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation = nn.GELU,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        self.activation = activation()


    def forward(
        self,
        tgt: Tensor,        # f2
        memory: Tensor,     # f1
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        else:
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderLayer(decoder_embed_dim, decoder_num_heads)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.decoder_frame_info = nn.Parameter(torch.zeros(2, 1, 1, decoder_embed_dim))
        self.norm_pix_loss = norm_pix_loss
        
        self.grid_size_h_w = (img_size//patch_size, img_size//patch_size) if isinstance(img_size, int) else (img_size[0]//patch_size, img_size[1]//patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], *self.grid_size_h_w, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], *self.grid_size_h_w, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_frame_info, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio is not None:
            # rec frame
            
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            return x, mask, ids_restore
        else:
            # keep frame
            
            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            return x
    
    def forward_decoder(self, x1, x2_latent, x2_mask, x2_ids_restore, x3_latent, x3_ids_restore):
        # embed tokens
        x1 = self.decoder_embed(x1)                                 # [2, 1+196, 768] -> [2, 1+196, 512]
        x2_latent = self.decoder_embed(x2_latent)                   # [2, 1+49, 768] -> [2, 1+49, 512]
        x3_latent = self.decoder_embed(x3_latent)                   # [2, 1+49, 768] -> [2, 1+49, 512]

        # for the subsequent cat
        B, x2_lt_L_, C = x2_latent.shape
    
        # append mask tokens to sequence
        # x2
        x2_mask_tokens = self.mask_token.repeat(x2_latent.shape[0], x2_ids_restore.shape[1] + 1 - x2_latent.shape[1], 1)            # [2, 147, 512]
        x2_ = torch.cat([x2_latent[:, 1:, :], x2_mask_tokens], dim=1)  # no cls token
        x2_ = torch.gather(x2_, dim=1, index=x2_ids_restore.unsqueeze(-1).repeat(1, 1, x2_latent.shape[2]))  # unshuffle
        x2 = torch.cat([x2_latent[:, :1, :], x2_], dim=1)  # append cls token                                          # [2, 1+196, 512]
        # x3
        x3_mask_tokens = self.mask_token.repeat(x3_latent.shape[0], x3_ids_restore.shape[1] + 1 - x3_latent.shape[1], 1)            # [2, 147, 512]
        x3_ = torch.cat([x3_latent[:, 1:, :], x3_mask_tokens], dim=1)  # no cls token
        x3_ = torch.gather(x3_, dim=1, index=x3_ids_restore.unsqueeze(-1).repeat(1, 1, x3_latent.shape[2]))  # unshuffle
        x3 = torch.cat([x3_latent[:, :1, :], x3_], dim=1)  # append cls token                                          # [2, 1+196, 512]
        
        # add pos embed
        x1 = x1 + self.decoder_pos_embed              # [2, 1+196, 512]
        x2 = x2 + self.decoder_pos_embed              # [2, 1+196, 512]
        x3 = x3 + self.decoder_pos_embed              # [2, 1+196, 512]
        
        # x2_latent for x3 extract infomation
        x2_latent_for_x3 = torch.cat( [x2[:, :1, :], x2[:, 1:, :][x2_mask.eq(0)].view(B, x2_lt_L_-1, C)], dim=1)    # CLS and x2_latent   
        x1_x2_for_x3 = torch.cat([x1 + self.decoder_frame_info[0], x2_latent_for_x3 + self.decoder_frame_info[1]], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x2 = blk(x2, x1 + self.decoder_frame_info[0])
        
        for blk in self.decoder_blocks:
            x3 = blk(x3, x1_x2_for_x3)
            
        x2 = self.decoder_norm(x2)                                    # [2, 197, 512]
        x3 = self.decoder_norm(x3)                                    # [2, 197, 512]

        # predictor projection
        x2 = self.decoder_pred(x2)                                    # [2, 197, 768]
        x3 = self.decoder_pred(x3)                                    # [2, 197, 768]

        # remove cls token
        x2 = x2[:, 1:, :]                                             # [2, 196, 768]
        x3 = x3[:, 1:, :]                                             # [2, 196, 768]

        return x2, x3

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep (no need to rec), 1 is remove (need to rec), 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, x1_imgs, x2_imgs, x3_imgs, mask_ratios=[0.95, 0.95], rec_weights=[0.8, 1.0]):
        x1 = self.forward_encoder(x1_imgs)
        x2_latent, x2_mask, x2_ids_restore = self.forward_encoder(x2_imgs, mask_ratio=mask_ratios[0])
        x3_latent, x3_mask, x3_ids_restore = self.forward_encoder(x3_imgs, mask_ratio=mask_ratios[1])
        
        x2_pred, x3_pred = self.forward_decoder(x1, x2_latent, x2_mask, x2_ids_restore, x3_latent, x3_ids_restore)  # [N, L, p*p*3]
        loss_x2 = self.forward_loss(x2_imgs, x2_pred, x2_mask)
        loss_x3 = self.forward_loss(x3_imgs, x3_pred, x3_mask)
        
        loss = rec_weights[0] * loss_x2 + rec_weights[1] * loss_x3
        return loss, loss_x2, loss_x3


def catmae_vit_small(patch_size=16, decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def catmae_vit_base(patch_size=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model






if "__main__" == __name__:
    img_size = 48
    model = catmae_vit_small(192,4,3, img_size=img_size)
    x1 = torch.randn(2,3,img_size,img_size)
    x2 = torch.randn(2,3,img_size,img_size)
    x3 = torch.randn(2,3,img_size,img_size)
    
    y,z,q = model(x1, x2, x3, mask_ratios=[0.75, 0.85])
    print(y, z, q)