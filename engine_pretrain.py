import math
import sys
from typing import Iterable

import torch
import time
import util.misc as misc
import util.lr_sched as lr_sched
from datetime import datetime

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    print("yeild data")
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print(formatted_time)
    for data_iter_step, (f1, f2, f3) in enumerate(metric_logger.log_every(data_loader, print_freq, header, args.log_dir)):
    # for data_iter_step, (f1, f2) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        f1 = f1.to(device, non_blocking=True)
        f2 = f2.to(device, non_blocking=True)
        f3 = f3.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, loss_x2, loss_x3  = model(f1, f2, f3, mask_ratios=args.mask_ratios, rec_weights=args.rec_weights)

        loss_value = loss.item()
        loss_x2_value = loss_x2.item()
        loss_x3_value = loss_x3.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_x2_value_reduce = misc.all_reduce_mean(loss_x2_value)
        loss_x3_value_reduce = misc.all_reduce_mean(loss_x3_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalars('train_loss_x2_x3', {"loss_x2":loss_x2_value_reduce, "loss_x3":loss_x3_value_reduce}, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}