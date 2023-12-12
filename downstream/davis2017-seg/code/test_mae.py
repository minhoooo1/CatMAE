from __future__ import print_function

import os
import time
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from model import CRW

from data import vos, jhmdb
from data.video import SingleVideoDataset

import utils
import utils.test_utils as test_utils
import models_mae
from tqdm import tqdm

def main(args, vis):
    #####################################################################################################
    # model = CRW(args, vis=vis).to(args.device)
    # args.mapScale = test_utils.infer_downscale(model)
    if "dino" in args.model_type:
        model = torch.hub.load('facebookresearch/dino:main', args.model_type)   # DINO
    else:
        model = models_mae.__dict__[args.model_type](ckpt_path=args.resume)
    args.mapScale = np.array([16, 16]) if "16" in args.model_type else np.array([8, 8])
    #####################################################################################################

    args.use_lab = args.model_type == 'uvc'
    dataset = (vos.VOSDataset if not 'jhmdb' in args.filelist  else jhmdb.JhmdbSet)(args)
    val_loader = torch.utils.data.DataLoader(dataset,
        batch_size=int(args.batchSize), shuffle=False, num_workers=args.workers, pin_memory=True)

    # cudnn.benchmark = False
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    #####################################################################################################
    # Load checkpoint.
    # if os.path.isfile(args.resume):
    #     print('==> Resuming from checkpoint..')
    #     checkpoint = torch.load(args.resume)
        
    #     if args.model_type == 'scratch':
    #         state = {}
    #         for k,v in checkpoint['model'].items():
    #             if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
    #                 state[k.replace('.1.weight', '.weight')] = v
    #             else:
    #                 state[k] = v
    #         utils.partial_load(state, model, skip_keys=['head'])
    #     else:
    #         utils.partial_load(checkpoint['model'], model, skip_keys=['head'])

    #     del checkpoint
    #####################################################################################################
    
    model.eval()
    model = model.to(args.device)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    with torch.no_grad():
        test_loss = test(val_loader, model, args)
            

def test(loader, model, args):
    n_context = args.videoLen
    D = None    # Radius mask
    
    for vid_idx, (imgs, imgs_orig, lbls, lbls_orig, lbl_map, meta) in tqdm(enumerate(loader)):
        t_vid = time.time()
        imgs = imgs.to(args.device)
        B, N = imgs.shape[:2]
        assert(B == 1)

        print('******* Vid %s (%s frames) *******' % (vid_idx, N))
        with torch.no_grad():
            t00 = time.time()

            ##################################################################
            # Compute image features (batched for memory efficiency)
            ##################################################################
            # bsize = 5   # minibatch size for computing features
            bsize = 3
            feats = []
            for b in range(0, imgs.shape[1], bsize):
                #####################################################################################################
                # feat = model.encoder(imgs[:, b:b+bsize].transpose(1,2).to(args.device))
                # imgs [1, 89, 3, 480, 880]
                # feat [B 1+L, C]  TODO  -> [1, C, B, GH, GW]
                feat = model(imgs[0, b:b+bsize].to(args.device))
                B, L_, C = feat.shape
                grid_HW = [30, 55] if "16" in args.model_type else [60, 110]
                feat = feat[:, 1:].view(B, *grid_HW, C).permute(3, 0, 1, 2)[None, ...]
                #####################################################################################################
                feats.append(feat.cpu())
            feats = torch.cat(feats, dim=2).squeeze(1)

            if not args.no_l2:
                feats = torch.nn.functional.normalize(feats, dim=1)

            print('computed features', time.time()-t00)

            if args.pca_vis and vis:
                pca_feats = [utils.visualize.pca_feats(feats[0].transpose(0, 1), K=1)]
                for pf in pca_feats:
                    pf = torch.nn.functional.interpolate(pf[::10], scale_factor=(4, 4), mode='bilinear')
                    vis.images(pf, nrow=2, env='main_pca')
                    import pdb; pdb.set_trace()

            ##################################################################
            # Compute affinities
            ##################################################################
            torch.cuda.empty_cache()
            t03 = time.time()
            
            # Prepare source (keys) and target (query) frame features
            key_indices = test_utils.context_index_bank(n_context, args.long_mem, N - n_context)
            key_indices = torch.cat(key_indices, dim=-1)           
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]

            # Make spatial radius mask TODO use torch.sparse
            restrict = utils.MaskedAttention(args.radius, flat=False)
            D = restrict.mask(*feats.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            print('computing affinity')
            Ws, Is = test_utils.mem_efficient_batched_affinity(query, keys, D, 
                        args.temperature, args.topk, args.long_mem, args.device)
            # Ws, Is = test_utils.batched_affinity(query, keys, D, 
            #             args.temperature, args.topk, args.long_mem, args.device)

            if torch.cuda.is_available():
                print(time.time()-t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024**2))

            ##################################################################
            # Propagate Labels and Save Predictions
            ###################################################################

            maps, keypts = [], []
            lbls[0, n_context:] *= 0 
            lbl_map, lbls = lbl_map[0], lbls[0]

            for t in range(key_indices.shape[0]):
                # Soft labels of source nodes
                ctx_lbls = lbls[key_indices[t]].to(args.device)
                ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

                # Weighted sum of top-k neighbours (Is is index, Ws is weight) 
                pred = (ctx_lbls[:, Is[t]] * Ws[t].to(args.device)[None]).sum(1)
                pred = pred.view(-1, *feats.shape[-2:])
                pred = pred.permute(1,2,0)
                
                if t > 0:
                    lbls[t + n_context] = pred
                else:
                    pred = lbls[0]
                    lbls[t + n_context] = pred

                if args.norm_mask:
                    pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                    pred[:, :, :] /= pred.max(-1)[0][:, :, None]

                # Save Predictions            
                cur_img = imgs_orig[0, t + n_context].permute(1,2,0).numpy() * 255
                _maps = []

                if 'jhmdb' in args.filelist.lower():
                    coords, pred_sharp = test_utils.process_pose(pred, lbl_map)
                    keypts.append(coords)
                    pose_map = utils.vis_pose(np.array(cur_img).copy(), coords.numpy() * args.mapScale[..., None])
                    _maps += [pose_map]

                if 'VIP' in args.filelist:
                    outpath = os.path.join(args.save_path, 'videos'+meta['img_paths'][t+n_context][0].split('videos')[-1])
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                else:
                    outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))

                heatmap, lblmap, heatmap_prob = test_utils.dump_predictions(
                    pred.cpu().numpy(),
                    lbl_map, cur_img, outpath)

                _maps += [heatmap, lblmap, heatmap_prob]
                maps.append(_maps)

                if args.visdom:
                    [vis.image(np.uint8(_m).transpose(2, 0, 1)) for _m in _maps]

            if len(keypts) > 0:
                coordpath = os.path.join(args.save_path, str(vid_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)
            
            if vis:
                wandb.log({'blend vid%s' % vid_idx: wandb.Video(
                    np.array([m[0] for m in maps]).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                wandb.log({'plain vid%s' % vid_idx: wandb.Video(
                    imgs_orig[0, n_context:].numpy(), fps=4, format="gif")})  
                
            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (vid_idx, time.time() - t_vid))


if __name__ == '__main__':
    args = utils.arguments.test_args()

    args.imgSize = args.cropSize
    print('Context Length:', args.videoLen, 'Image Size:', args.imgSize)
    print('Arguments', args)

    vis = None
    if args.visdom:
        import visdom
        import wandb
        vis = visdom.Visdom(server=args.visdom_server, port=8095, env='main_davis_viz1'); vis.close()
        wandb.init(project='palindromes', group='test_online')
        vis.close()

    main(args, vis)
