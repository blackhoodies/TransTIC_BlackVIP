# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from cmath import exp
import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from losses import *
import yaml

from utils.advance_models import wCoordinator
from utils.tools import load_pretrained_weights

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Set optimizer for only the parameters for propmts"""

    if args.TRANSFER_TYPE == "prompt":
        parameters = {
            k
            for k, p in net.named_parameters()
            if "prompt" in k
        }
    elif args.TRANSFER_TYPE == "BlackVIP":
        parameters = {
            k
            for k, p in net.named_parameters()
            if "coordinator" in k.split('.')[0] and "dec" in k.split('.')[1]
        }
    else:
        raise NotImplementedError

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer


def train_one_epoch(
    model, approx_loss, train_dataloader, optimizer, lmbda, step
):
    b1 = 0.9
    a = 0.01
    c = 0.01
    gamma = 0.1
    alpha = 0.4
    o = 1.0
    model.train()
    device = next(model.parameters()).device
    tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
    for i, (d,l) in tqdm_emu:
        d = d.to(device)
        l = l.to(device)
        ak = a/((step + o)**alpha)
        ck = c/(step**gamma)

        optimizer.zero_grad()

        out_net = model(d)
        
        # SPSA-GC
        w_enc = torch.nn.utils.parameters_to_vector(model.coordinator_enc.dec.parameters())

        ghat, total_loss, accu, out_criterion, perc_loss, loss, model = approx_loss.spsa_grad_estimate_bi(w_enc, model, d, l, lmbda, ck)
        if step > 1:  
            m1 = b1*m1 + ghat
        else:              
            m1 = ghat
        accum_ghat = ghat + b1*m1
         

        #* param update
        w_new_enc = w_enc - ak * accum_ghat
        torch.nn.utils.vector_to_parameters(w_new_enc, model.coordinator_enc.dec.parameters())

        w_dec = torch.nn.utils.parameters_to_vector(model.coordinator_dec.dec.parameters())
        ghat, total_loss, accu, out_criterion, perc_loss, loss, model = approx_loss.spsa_grad_estimate_bi(w_dec, model, d, l, lmbda, ck)
        if step > 1:  
            m1 = b1*m1 + ghat
        else:              
            m1 = ghat
        accum_ghat = ghat + b1*m1
         

        #* param update
        w_new_dec = w_dec - ak * accum_ghat
        torch.nn.utils.vector_to_parameters(w_new_dec, model.coordinator_dec.dec.parameters())
        # total_loss.backward()
        # optimizer.step()

        update_txt=f'[{i*len(d)}/{len(train_dataloader.dataset)}] | Loss: {total_loss.item():.3f} | MSE loss: {out_criterion["mse_loss"].item():.5f} | Bpp loss: {out_criterion["bpp_loss"].item():.4f}'
        tqdm_emu.set_postfix_str(update_txt, refresh=True)
        step += 1

def test_epoch(epoch, test_dataloader, model, criterion_rd, criterion_cls, lmbda, stage='test'):
    model.eval()
    device = next(model.parameters()).device

    loss_am = AverageMeter()
    percloss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    accuracy = AverageMeter()
    totalloss = AverageMeter()

    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader))
        for i, (d,l) in tqdm_meter:
            d = d.to(device)
            l = l.to(device)
            out_net = model(d)
            out_criterion = criterion_rd(out_net, d)
            loss, accu, perc_loss = criterion_cls(out_net, d, l)
            total_loss = 1000*lmbda*perc_loss + out_criterion['bpp_loss']

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss_am.update(loss)
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            accuracy.update(accu)
            percloss.update(perc_loss)
            totalloss.update(total_loss)

    txt = f"Loss: {loss_am.avg:.3f} | MSE loss: {mse_loss.avg:.5f} | Bpp loss: {bpp_loss.avg:.4f} | accu: {accuracy.avg:.4f}\n"
    tqdm_meter.set_postfix_str(txt)

    model.train()
    print(f"{epoch} | bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f} | accu: {accuracy.avg:.5f}")
    return loss_am.avg

def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/vpt_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)
    
    parser.add_argument(
        "-T",
        "--TEST",
        action='store_true',
        help='Testing'
    )

    args = parser.parse_args(remaining)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    cls_transforms = transforms.Compose(
        [transforms.Resize(args.patch_size), transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    if args.dataset=='imagenet':
        train_dataset = torchvision.datasets.ImageNet(args.dataset_path,split='train', transform=cls_transforms)
        test_dataset = torchvision.datasets.ImageNet(args.dataset_path,split='val', transform=cls_transforms)
        val_dataset,_ = torch.utils.data.random_split(test_dataset,[15000,35000])
        small_train_datasets = torch.utils.data.random_split(train_dataset,[40000]*32+[1167])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    val_dataloader = DataLoader(val_dataset,batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    test_dataloader = DataLoader(test_dataset,batch_size=args.test_batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
  
    net = image_models[args.model](quality=int(args.quality_level), prompt_config=args.prompt_config)
    net = wCoordinator(args, net)
    
    net = net.to(device)
    for param in net.parameters():
        param.requires_grad = False
    if args.TRANSFER_TYPE == "prompt":
        for k, p in net.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False

    # if args.MODEL.INIT_WEIGHTS:
    #     load_pretrained_weights(net.coordinator.dec, args.MODEL.INIT_WEIGHTS)

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.5)
    rdcriterion = RateDistortionLoss(lmbda=args.lmbda)
    clscriterion = Clsloss(device, True)
    approx_loss = Loss(model=net, device=device, lmbda=args.lmbda, perceptual_loss=True)

    last_epoch = 0
    if args.checkpoint: 
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                if args.restore == 'scratch':
                    name = f"net.{k[7:]}"
                else:
                    name = k[7:] 
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        net.load_state_dict(new_state_dict, strict=True if args.TEST else False)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    
    if args.TEST:
        best_loss = float("inf")
        tqrange = tqdm.trange(last_epoch, args.epochs)
        loss = test_epoch(-1, test_dataloader, net, rdcriterion, clscriterion, args.VPT_lmbda, 'test')
        return

    best_loss = float("inf")
    tqrange = tqdm.trange(last_epoch, args.epochs)
    # loss = test_epoch(-1, val_dataloader, net, rd_criterion, criterion_cls, args.VPT_lmbda,'val')
    for epoch in tqrange:
        train_dataloader = DataLoader(
            small_train_datasets[epoch%32],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
        )
        step = args.batch_size*epoch + 1
        train_one_epoch(
            net,
            approx_loss,
            train_dataloader,
            optimizer,
            args.VPT_lmbda, 
            step
        )
        loss = test_epoch(epoch, val_dataloader, net, rdcriterion, clscriterion, args.VPT_lmbda, 'val')
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir,
                filename=f'checkpoint_{epoch}.pth.tar'
            )
            if epoch%10==9:
                shutil.copyfile(base_dir+'checkpoint.pth.tar', base_dir+ f"checkpoint_{epoch}.pth.tar" )
    


if __name__ == "__main__":
    # import compressai
    # print(sys.path)
    # print(os.path.abspath(compressai.__file__))
    main(sys.argv[1:])
0