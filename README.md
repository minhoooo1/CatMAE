## Concatenated Masked Autoencoders as Spatial-Temporal Learner: A PyTorch Implementation

<p align="center">
<img src="https://github.com/minhoooo1/CatMAE/blob/master/figures/arch.png" width="800">
</p>

This is a PyTorch re-implementation of the paper [Concatenated Masked Autoencoders as Spatial-Temporal Learner](https://arxiv.org/abs/2311.00961):


##  Requirements
- pytorch (2.0.1)
- [`timm==0.4.12`](https://github.com/rwightman/pytorch-image-models)
- decord


## Data Preparation
We use two datasets, [Kinetics-400](https://deepmind.com/research/open-source/kinetics) and [DAVIS-2017](https://davischallenge.org/davis2017/code.html), for training and downstream tasks in total.

- **Kinetics-400** used in our experiment comes from [here](https://opendatalab.com/Kinetics-400).
- **DAVIS-2017** used in our experiment comes from [here](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)

## Pre-training

To pre-train Cat-ViT-Small (recommended default), run the following commond:

```
bash pretrain.sh configs/pretrain_catmae_vit-s-16.json
```

Some important arguments
- The `data_path` is /path/to/Kinetics-400/videos_train/
- The effective batch size is `batch_size` (256) * num of `gpus` (4) * `accum_iter` (2) = 2048
- The effective epochs is `epochs` (150) * `repeated_sampling` （2） = 300
- The default `model` is **catmae_vit_small** (with default patch_size and decoder_dim_dep_head), and for training VIT-B, you can alse change it to **catmae_vit_base**.
- Here we use `--norm_pix_loss` as the target for better representation learning.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective_batch_size / 256. 


## Pre-trained checkpoints
The following table provides the pre-trained checkpoints used in the paper
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT/16-Small</th>
<th valign="bottom">ViT/8-Small</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1xWrpSxZy6d3r_XnsZmXvqM1XUReJ7v97/view?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1ksYZJPa2pZ-NYWjYKLh05-bt_A40Rhm7/view?usp=drive_link">download</a></td>
</tr>
<tr><td align="left">DAVIS 2017 J&Fm</td>
<td align="center">62.5</td>
<td align="center">70.4</td>
</tr>

</tbody></table>


## Video segment in DAVIS-2017

The Video segment instruction is in [DAVIS.md](downstream/davis2017-seg/DAVIS.md).

## Action recognition in Kinetics-400

### Coming Soon
