
import time
import argparse
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from models import INN
from dataset import ValDataset
from dataloaders import train_dali_loader,dali_loader,image_pipeline
from utils import svd_orthogonalization, close_logger, init_logging, normalize_augment,open_image
from train_common import resume_training, lr_scheduler, log_train_psnr, \
                    validate_and_log, save_model_checkpoint
import timm
import timm.optim
import timm.scheduler

import random
import os
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], mask=None, return_feature=False):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                if mask is not None:
                    _, _, H, W = x.shape
                    mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')[:, 0:1, :, :]
                    x = x * mask_resized
                    y = y * mask_resized
                    loss += torch.nn.functional.l1_loss(x, y)
                else:
                    loss += torch.nn.functional.l1_loss(x, y)

                if return_feature:
                    return x, y

            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')


    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))


def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    # if im.ndim == 3:
    #     im = im.transpose((2, 0, 1))
    # else:
    #     im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)

def markloss(train,val):
    f1 = open('train_loss.txt','a')
    f2 = open('val_loss.txt','a')
    f1.write(str(train)+'\n')
    #f2.write(str(val)+'\n')
    f1.close()
    f2.close()
 
def main(**args):
    r"""Performs the main training loop
    """

    # Load dataset
    print('> Loading datasets ...')
    dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
    """ loader_train = train_dali_loader(batch_size=args['batch_size'],\
                                    file_root=args['trainset_dir'],\
                                    sequence_length=args['temp_patch_size'],\
                                    crop_size=args['patch_size'],\
                                    epoch_size=args['max_number_patches'],\
                                    random_shuffle=True,\
                                    temp_stride=3) """
    
    num_minibatches = int(args['max_number_patches']//args['batch_size'])
    ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
    print("\t# of training samples: %d\n" % int(args['max_number_patches']))

    # Init loggers
    writer, logger = init_logging(args)

    # Define GPU devices
    device_ids = [0]
    torch.backends.cudnn.benchmark = True # CUDNN optimization

    # Create model
    model = INN()
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    print('# of Generator parameters: %s', sum(param.numel() for param in model.parameters()))
    total = sum([param.nelement() for param in model.parameters()])
    print("##########Number of parameter: %.2fM" % (total/1e6))
    # Define loss
    L1loss = nn.L1Loss()
    L1loss.cuda()
    vggloss = VGGPerceptualLoss()
    vggloss.cuda()
    L1loss2 = nn.L1Loss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # Resume training or start anew
    start_epoch, training_params = resume_training(args, model, optimizer)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,t_initial=25,lr_min=1e-5,warmup_t=2,warmup_lr_init=1e-5)
    # Training
    start_time = time.time()
    
    arr = np.loadtxt('noise.txt')
    
    for epoch in range(start_epoch, args['epochs']):
        # Set learning rateDAVISDAVIS
        #scheduler.step(6)
        
        current_lr = 1e-4
        
        #current_lr, reset_orthog = lr_scheduler(epoch, args)
        #if reset_orthog:
        #	training_params['no_orthog'] = True

        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('\nlearning rate %f' % current_lr)

        # train
        loss_epoch = 0
        for i in range(128000):
            
            seq_list = []
            for j in range(5):
                files = "./train/train_data/{}_{}.png".format(i,j)

                img, expanded_h, expanded_w = open_image(files,\
                                                        gray_mode=False,\
                                                        expand_if_needed=False,\
                                                        expand_axis0=False)
                seq_list.append(torch.from_numpy(img))
            img_train = torch.cat(seq_list, dim=0)
            img_train = img_train.unsqueeze(dim=0).cuda(non_blocking=True)
            gt_train = img_train[:,6:9,:,:]
            # Pre-training step
            model.train()
            
            # When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
            optimizer.zero_grad()
            N, _, H, W = img_train.size()

            noise = torch.empty_like(img_train).normal_(mean=0, std=arr[i] / 255.).cuda(non_blocking=True)
            noise = noise.cuda(non_blocking=True)
            #define noisy input
            imgn_train = img_train + noise

            # Send tensors to GPU
            gt_train = gt_train.cuda(non_blocking=True)
            imgn_train = imgn_train.cuda(non_blocking=True)
            
            sigma = noise_estimate(imgn_train[:, 6:9, :, :].squeeze().cpu().numpy()) * 255#arr[i]
            
            noise_map = torch.randn(imgn_train[:, 0:1, :, :].size()).cuda().mul_(sigma / 255.0)
            
            # Evaluate model and optimize it
            out_train = model(imgn_train, noise_map)

            loss = 0.5*vggloss(gt_train, out_train) + L1loss(gt_train, out_train)
            loss_epoch+=loss.item()
            loss.backward()
            optimizer.step()

            # Results
            if training_params['step'] % args['save_every'] == 0:
                # Apply regularization by orthogonalizing filters
                # if not training_params['no_orthog']:
                # 	model.apply(svd_orthogonalization)

                # Compute training PSNR
                log_train_psnr(out_train, \
                                gt_train, \
                                loss, \
                                writer, \
                                epoch, \
                                i, \
                                num_minibatches, \
                                training_params, sigma)
            # update step counter
            training_params['step'] += 1

        # Call to model.eval() to correctly set the BN layers before inference
        model.eval()
        loss_epoch /= 128000
        # Validation and log images
        psnr,valloss = validate_and_log(
                        model_temp=model, \
                        dataset_val=dataset_val, \
                        valnoisestd=args['val_noiseL'], \
                        temp_psz=args['temp_patch_size'], \
                        writer=writer, \
                        epoch=epoch, \
                        lr=current_lr, \
                        logger=logger, \
                        trainimg=img_train,\
                        vgg=vggloss,\
                        l1=L1loss2
                        )

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        markloss(loss_epoch,valloss)
        save_model_checkpoint(model, args, optimizer, training_params, epoch, psnr)

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    # Close logger file
    close_logger(logger)

def seed_torch(seed = 3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the denoiser")

    #Training parameters
    parser.add_argument("--batch_size", type=int, default=1, 	\
                     help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=80, \
                     help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true',\
                        help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[400, 500], \
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=0.00005, \
                     help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true',\
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=100,\
                        help="Number of training steps to log psnr and perform \
                        orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=1,\
                        help="Number of training epochs to save state")
    parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
                     help="Noise training interval")
    parser.add_argument("--val_noiseL", type=float, default=10, \
                        help='noise level used on validation set')
    # Preprocessing parameters
    parser.add_argument("--patch_size", "--p", type=int, default=128, help="Patch size")
    parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
    parser.add_argument("--max_number_patches", "--m", type=int, default=128000, \
                        help="Maximum number of patches")
    # Dirs
    parser.add_argument("--log_dir", type=str, default="logs", \
                     help='path of log files')
    parser.add_argument("--trainset_dir", type=str, default="./data", \
                     help='path of trainset')
    parser.add_argument("--valset_dir", type=str, default="./Set8", \
                         help='path of validation set')
    argspar = parser.parse_args()

    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noise_ival[0] /= 255.
    argspar.noise_ival[1] /= 255.

    print("\n### Training denoiser model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    seed_torch()
    main(**vars(argspar))
