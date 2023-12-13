<h1 style='font-size: 1.6em'>Action Recognition Finetune for Kinetics-400</h1>

The action recognition finetuning is based on the repository [videoMAE](https://github.com/MCG-NJU/VideoMAE).
##  Requirements
- pytorch (We recommend you to use PyTorch >= 1.8.0)
- timm==0.4.12
- decord
- einops

## Data
Please follow the instructions in [this](https://github.com/MCG-NJU/VideoMAE/blob/main/DATASET.md) for data preparation.
- **Recommend**: [OpenDataLab](https://opendatalab.com/) provides a copy of [Kinetics400](https://opendatalab.com/Kinetics-400) dataset, you can download Kinetics dataset with **short edge 320px** from [here](https://opendatalab.com/Kinetics-400).
- Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv` ( here `test.csv` is the same as `val.csv`). The format of `*.csv` file is like:

```
dataset_root/video_1.mp4  label_1
dataset_root/video_2.mp4  label_2
dataset_root/video_3.mp4  label_3
...
dataset_root/video_N.mp4  label_N
```

## Pretrained Model
The following table provides the pre-trained checkpoints used in the paper
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">CatMAE-ViT/16-Small</th>
<th valign="bottom">VideoMAE-ViT/16-Small</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://drive.google.com/file/d/1xWrpSxZy6d3r_XnsZmXvqM1XUReJ7v97/view?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1nU-H1u3eJ-VuyCveU7v-WIOcAVxs5Hww/view?usp=sharing">download</a></td>
</tr>
<tr><td align="left">pretrain epoch</td>
<td align="center">300</a></td>
<td align="center">1600</a></td>
</tr>
</tr>
<tr><td align="left">Top-1</td>
<td align="center">-</a></td>
<td align="center">-</a></td>
</tr>
<tr><td align="left">Top-5</td>
<td align="center">-</a></td>
<td align="center">-</a></td>
</tr>


</tbody></table>

## Finetune
We finetune model for 150 epochs. Before running following commands, remember to download pretrained weight.

**CatMAE-ViT/16-Small**
```
cd downstream/kinetics-400-action-recognition
bash scripts/kinetics/2d_patch_vit_small_patch16_224/finetune.sh
```

**VideoMAE-ViT/16-Small**
```
bash scripts/kinetics/2d_patch_vit_small_patch16_224/finetune.sh
```
