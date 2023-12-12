<h1 style='font-size: 1.6em'>Label Propagation for DAVIS 2017 Semi-supervised Video Segmentation</h1>

<!-- ![](https://github.com/ajabri/videowalk/raw/master/figs/teaser_animation.gif) -->
<p align="center">
<img src="figs/label_propagation.gif" width="600">
</p>

The label propagation algorithm is based on the repository [videowalk](https://github.com/ajabri/videowalk).

Propagating the ground truth from the first frame to subsequent frames in the video.

##  Requirements
- pytorch
- cv2
- matplotlib
- skimage
- imageio


## Data
```
bash download_process_data.sh
```
the script performs the following operations:
- Prepare the data by downloading the [Semi-supervised 2017 TrainVal 480p](https://davischallenge.org/davis2017/code.html).
- Resize the image to [480, 880] in order to ensure that it is divisible by the `patch_size` in ViT.
- In `davis_vallist_480_880.txt`, we have provided the paths to the images to be evaluated.



### Pretrained Model
You need to prepare the model weights for evaluation.

|  Method  |  Backbone  | Model-type | Resume (weight) | 
| :------: | :--------: | :---: | :---: |
| Surpervised |  ResNet-18  |  imagenet18  |  ***No need to download***  |
| Surpervised |  ResNet-50  |  imagenet50  |  ***No need to download***  |
| SimSiam   |  ResNet-50    |  simsiam     |  [simsiam](https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar)  |
| MoCo-v2 |  ResNet-50      |  mocov2      |  [mocov2](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar)  |
| DINO |  ResNet-50         |  dino_r50    |  [dino_r50](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth)  |
| VFS |  ResNet-50          |  vfs         |  [vfs](https://github.com/xvjiarui/VFS/releases/download/v0.1-rc1/r50_nc_sgd_cos_100e_r5_1xNx2_k400-d7ce3ad0.pth)  |
| CRW |  ResNet-18          |  crw         |  [crw](https://github.com/ajabri/videowalk/blob/master/pretrained.pth)  |
| MAE |  ViT-B/16           |  vitb16      |  [mae-vit-b-16](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)  |
| MAE-ST |  ViT-B/2x16x16   | -            |  ***-***  |
| VideoMAE |  ViT-B/2x16x16 | -            |  ***-***  |
| DINO |  ViT-S/16          |  dino_vits16 |  ***No need to download***  |
| SiamMAE |  ViT-S/16       |  -      |  ***-***  |
| CatMAE  |  ViT-S/16       |  -      |  ***-***  |
| CatMAE-ViRe  |  ViT-S/16  |  vits16      |  [catmae-vit-s-16](https://drive.google.com/file/d/1QUFeeb8U-WoDvOkI768R7B8BpmXw-jkg/view?usp=drive_link)  |
| DINO |  ViT-S/8           |  dino_vits8  |  ***No need to download***  |
| SiamMAE |  ViT-S/8        |  -       |  ***-***  |
| CatMAE  |  ViT-S/8        |  -       |  ***-***  |
| CatMAE-ViRe  |  ViT-S/8   |  vits8       |  [catmae-vit-s-8](https://drive.google.com/file/d/1jaYjnXYOJFSVoXfQWTJfPix6tddQ4YMF/view?usp=drive_link)  |



---

## Evaluation: Label Propagation
The label propagation algorithm is described in `test.py` and `test_mae.py`.  The output must be **post-processed** for evaluation.



### Label Propagation
It takes some time.

**resnet model**
```
python code/test.py --model-type imagenet18
--save-path results/imagenet18/T07_K8_R18_V20_480_880 \
--temperature 0.7 --topk 8 --radius 18 --videoLen 20 \
--filelist davis_vallist_480_880.txt
```

**vit model**
```
python code/test_mae.py --model-type vitb16 --resume mae_pretrain_vit_base.pth
--save-path results/mae-vitb/T07_K8_R18_V20_480_880 \
--temperature 0.7 --topk 8 --radius 18 --videoLen 20 \
--filelist davis_vallist_480_880.txt
```

- According to the table above `model-type` and `resume` are corresponding. For the **Resume** marked as **No need to download**, there is no need to pass in `--resume`. For other models, you need to do so. For example, `--model-type simsiam --resume simsiam_checkpoint_0099.pth.tar`. (Remember to download first)
- `save-path` is the path where the label-propagation data will be saved.
- The parameters `temperature`, `topk`, `radius`, and `videoLen` will affect the pixel similarity on different frame features based on the nearest neighbor method. For the weights of vit-s-8, the `temperature`, `topk`, `radius`, and `videoLen` are set to 0.5, 12, 32, and 20, respectively. For other weights, we use 0.7, 8, 18, and 20.
- The file `davis_vallist_480_880.txt` records the list of image to be propagated.

### Post-Process
It takes some time.

```
# Convert
python code/eval/convert_davis.py --in_folder results/imagenet18/T07_K8_R18_V20_480_880/ --out_folder results/imagenet18/T07_K8_R18_V20_480_880_out/ --dataset data/DAVIS_480_880/
```

- `in_folder` comes from `save-path` in test.py or test_mae.py
- `out_folder` is the path where the post-processed data will be saved.

### Evaluation
To evaluate a trained model on the DAVIS task, clone the [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) repository. Follow the instructions in this repository to install the **required environment** for this library.
```
# Compute metrics
python /path/to/davis2017-evaluation/evaluation_method.py \
--davis_path /path/to/davis2017-seg/data/DAVIS_480_880
--set val --task semi-supervised  
--results_path /path/to/results/imagenet18/T07_K8_R18_V20_480_880_out/  \
```

- `results_path` comes from `out_folder` in convert_davis.py

## Citation

Please cite.

```latex
@article{jiang2023concatenated,
  title={Concatenated Masked Autoencoders as Spatial-Temporal Learner},
  author={Jiang, Zhouqiang and Wang, Bowen and Xiang, Tong and Niu, Zhaofeng and Tang, Hong and Li, Guangshun and Li, Liangzhi},
  journal={arXiv preprint arXiv:2311.00961},
  year={2023}
}
```

```latex
@inproceedings{jabri2020walk,
    Author = {Allan Jabri and Andrew Owens and Alexei A. Efros},
    Title = {Space-Time Correspondence as a Contrastive Random Walk},
    Booktitle = {Advances in Neural Information Processing Systems},
    Year = {2020},
}
```
