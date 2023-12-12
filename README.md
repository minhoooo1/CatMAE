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

### Coming Soon


## Video segment in DAVIS-2017

### Coming Soon

## Action recognition in Kinetics-400

### Coming Soon
