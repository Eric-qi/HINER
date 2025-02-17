import os
import csv
import sys
import time
import random
import shutil
import imageio
import argparse
import numpy as np
from copy import deepcopy
from datetime import datetime
from dahuffman import HuffmanCodec
# from pygifsicle import optimize
from timm.models.layers.helpers import to_2tuple

from scipy.io import loadmat
from scipy.io import savemat
from matplotlib import colors
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.utils.data as Data
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


from spectralformer_utils import *
from hiner_utils import *
from data import HSIDataSet
from models import ViT, HINER, HINERDecoder, TransformInput

# torch.autograd.set_detect_anomaly(True)
def parse_args(argv):
    parser = argparse.ArgumentParser(description='INR-based Compression and Classification')
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='', help='data path for vid')
    parser.add_argument('--vid', type=str, default='k400_train0', help='hsi id',)
    parser.add_argument('--shuffle', action='store_true', help='randomly shuffle the mapping between frame idx and frame')
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, the first 19 samples is full train set, and the first 18 samples is chose currently')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='hsi crop size',)
    parser.add_argument('--ori_shape', type=str, default='640_1280', help='hsi ori size',)
    parser.add_argument('--resize_list', type=str, default='-1', help='hsi resize size',)

    # NERV architecture parameters
    # Embedding and encoding parameters
    parser.add_argument('--arch', type=str, default='hiner', help='model architecture')
    parser.add_argument('--data_type', type=str, default='hsi', help='datasets architecture')
    parser.add_argument('--embed', type=str, default='', help='empty string for HINER, and base value/embed_length for NeRV position encoding')
    parser.add_argument('--ks', type=str, default='0_3_3', help='kernel size for encoder and decoder')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[], help='stride list for encoder')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='enc latent dim and embedding ratio')
    parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
    parser.add_argument('--saturate_stages', type=int, default=-1, help='saturate stages for model size computation')

    # Decoding parameters: FC + Conv
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
    parser.add_argument('--reduce', type=float, default=1.2, help='chanel reduction for next stage')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list for decoder')
    parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel'], type=str, nargs="+",
        help='conv type for encoder/decoder', choices=['none', 'pshuffel', 'conv', 'convnext', 'interpolate'])
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', 
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Epoch number')
    parser.add_argument('--block_params', type=str, default='1_1', help='residual blocks and percentile to save')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
    parser.add_argument('--loss', type=str, default='Fusion6', help='loss type, default=L2')
    parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for model quantization')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='bit length for embedding quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    parser.add_argument('--encoder_file',  default='', type=str, help='specify the embedding file')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--weight_class', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--weight_header', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    # classification
    parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset to use')
    parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
    parser.add_argument('--mode', choices=['ViT', 'CAF', 'Super'], default='ViT', help='mode choice')
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
    parser.add_argument('--patches', type=int, default=1, help='number of patches')
    parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
    parser.add_argument('--epoches', type=int, default=300, help='epoch number')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

    # hyper parameters
    parser.add_argument('--seed', help='seed number', default=0, type=int)

    args = parser.parse_args(argv)

    return args


class ScalingNet(nn.Module):
    def __init__(self, channel=200, squeeze_factor=2):
        super(ScalingNet, self).__init__()
        self.channel = int(channel)

        self.fc1 = nn.Linear(1, channel//squeeze_factor, bias=True)
        self.fc2 = nn.Linear(channel//squeeze_factor, channel, bias=True)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x, lambda_rd=0.002):
        b, c, _, _ = x.size()
        # scaling_vector = torch.exp(10 * self.fc2(F.gelu(self.fc1(lambda_rd))))
        # scaling_vector = self.fc2(F.gelu(self.fc1(lambda_rd)))
        scaling_vector = self.fc2(F.gelu(self.fc1(lambda_rd)))
        scaling_vector = scaling_vector.view(b, c, 1, 1)
        x_scaled = x * scaling_vector.expand_as(x)
        return x_scaled

class ConvMlp(nn.Module):
    def __init__(
            self, in_features=200, hidden_features=100, out_features=200, act_layer=nn.GELU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Header(nn.Module):
    def __init__(
            self, in_features=200, squeeze_factor=2, hidden_features=50, out_features=200, act_layer=nn.GELU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()

        self.scale = ScalingNet(channel=in_features, squeeze_factor=squeeze_factor)
        self.mlp = ConvMlp(in_features, hidden_features, out_features, 
                           act_layer, norm_layer, bias, drop)

    def forward(self, x, lambda_rd=0.2):
        shortcut = x
        x = self.scale(x, lambda_rd)
        x = torch.transpose(x + shortcut, 0, 1) 

        x1 = self.mlp(x)
        x = torch.transpose(x1 + x, 0, 1)
        x = x + shortcut

        return x


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def data_to_gpu(x, device):
    return x.to(device)

def quant_model(model, args):
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]
        encoder_k_list = []
        for k,v in cur_ckt.items():
            if 'encoder' in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = quant_tensor(v, args.quant_model_bit)
                quant_ckt[k] = quant_v
                cur_ckt[k] = new_v
        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        
        return model_list, quant_ckt


# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        print(batch_idx)
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        # batch_data = torch.autograd.Variable(batch_data, requires_grad=True)
        # optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()      

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)
        
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre

def train(local_rank, args):

    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)

    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim',
        'quant_seen_psnr', 'quant_seen_ssim', 'quant_unseen_psnr', 'quant_unseen_ssim']
    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]

    # setup dataloader for coder
    full_dataset = HSIDataSet(args)
    sampler = torch.utils.data.distributed.DistributedSampler(full_dataset) if args.distributed else None
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batchSize, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=False, worker_init_fn=worker_init_fn)   
    
    args.final_size = full_dataset.final_size
    args.full_data_length = len(full_dataset)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    train_ind_list, args.val_ind_list = data_split(list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0)
    args.dump_vis = args.dump_images

    # setup dataloader for classifier
    data = loadmat(args.data_path)
    color_mat = loadmat('./data/AVIRIS_colormap.mat')
    TR = data['TR']
    TE = data['TE']
    input = data['input'] #(145,145,200)
    label = TR + TE
    num_classes = np.max(TR)
    color_mat_list = list(color_mat)
    color_matrix = color_mat[color_mat_list[3]] #(17,3)
    # normalize data by band norm
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    height, width, band = input.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    #-------------------------------------------------------------------------------
    # obtain train and test data
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
    # load data
    x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    Label_train=Data.TensorDataset(x_train,y_train)
    x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
    Label_test=Data.TensorDataset(x_test,y_test)
    x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    Label_true=Data.TensorDataset(x_true,y_true)

    # label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)

    #  Make sure the testing dataset is fixed for every run
    train_dataset =  Subset(full_dataset, train_ind_list)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # Compute the parameter number
    if 'pe' in args.embed or 'le' in args.embed:
        embed_param = 0
        embed_dim = int(args.embed.split('_')[-1]) * 2
        fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
    else:
        total_enc_strds = np.prod(args.enc_strds)
        embed_hw = args.final_size / total_enc_strds**2
        enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
        embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
        embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.full_data_length
        args.enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
        fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

    decoder_size = args.modelsize * 1e6 - embed_param
    ch_reduce = 1. / args.reduce
    dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
    fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
    a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
    b =  embed_dim * fc_param 
    c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
    args.fc_dim = int(np.roots([a,b,c - decoder_size]).max())

    # Building model
    
    coder = HINER(args)
    checkpoint = torch.load(args.weight, map_location='cpu')
    coder.load_state_dict(checkpoint['state_dict'])
    
    header = Header(in_features=band, 
                    squeeze_factor=2,
                    hidden_features=50, 
                    out_features=band, 
                    act_layer=nn.GELU,
                    # norm_layer=nn.LayerNorm,
                    drop=0.)
    # checkpoint_header = torch.load(args.weight_header, map_location='cpu')
    # header.load_state_dict(checkpoint_header)
    
    classifier = ViT(
    image_size = args.patches,
    near_band = args.band_patches,
    num_patches = band,
    num_classes = num_classes,
    dim = 64,
    depth = 5,
    heads = 4,
    mlp_dim = 8,
    dropout = 0.1,
    emb_dropout = 0.1,
    mode = args.mode
    )
    # checkpoint_class = torch.load(args.weight_class, map_location='cpu')
    # classifier.load_state_dict(checkpoint_class)

    beta = torch.Tensor([2e-5]).cuda()

    ##### get model params and flops #####
    if local_rank in [0, None]:
        encoder_param = (sum([p.data.nelement() for p in coder.encoder.parameters()]) / 1e6) 
        decoder_param = (sum([p.data.nelement() for p in coder.decoder.parameters()]) / 1e6) 
        total_param = decoder_param + embed_param / 1e6
        args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
        param_str = f'Encoder_{round(encoder_param, 2)}M_Decoder_{round(decoder_param, 2)}M_Total_{round(total_param, 2)}M'
        print(f'{args}\n {classifier}\n {header}\n {coder}\n {param_str}', flush=True)
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(classifier) + str(header) + str(coder) + '\n' + f'{param_str}\n')
        writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        coder = torch.nn.parallel.DistributedDataParallel(coder.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        coder = torch.nn.DataParallel(coder)
    elif torch.cuda.is_available():
        coder = coder.cuda()
        header = header.cuda()
        classifier = classifier.cuda()
    

    # coder_optimizer = optim.Adam(coder.parameters(), weight_decay=0.)
    args.transform_func = TransformInput(args)
    criterion = nn.CrossEntropyLoss().cuda()
    classifier_optimizer = torch.optim.Adam(list(header.parameters())+list(classifier.parameters()), 
                                            lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.eval_only:
        print_str = 'Evaluation ... \n {} Results for checkpoint: {}\n'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), args.weight)
        # results_list, hw, input_normalize_hat = evaluate(coder, full_dataloader, local_rank, args, args.dump_vis, huffman_coding=True)
        results_list, hw = evaluate(coder, full_dataloader, local_rank, args, args.dump_vis, huffman_coding=True)
        print_str = f'PSNR for output {hw} for quant {args.quant_str}: '

        rec = header(input_normalize_hat, lambda_rd=beta)

        input_normalize_hat = crop(rec, height, width)
        input_normalize_hat = torch.squeeze(input_normalize_hat, dim=1).permute(1, 2, 0)

        # device = next(coder.parameters()).device
        x_test_band_hat = train_hat(input_normalize_hat, height, width, band, total_pos_test, patch=args.patches, band_patch=args.band_patches)
        x_test_hat = x_test_band_hat.permute(0,2,1) #[695, 200, 7, 7]
        # label_test_loader = samples(x_test_hat, y_test, args.batch_size, shuffle=True, generator=generator)
        Label_test=Data.TensorDataset(x_test_hat,y_test)
        label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)

        tar_v, pre_v = valid_epoch(classifier, label_test_loader, criterion, classifier_optimizer)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

        for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
            best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
            cur_v = RoundTensor(best_metric_value, 2 if 'psnr' in metric_name else 4)
            print_str += f'best_{metric_name}: {cur_v} | '
            best_metric_list[i] = best_metric_value
        if local_rank in [0, None]:
            print(print_str, flush=True)
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n\n')        
            args.train_time, args.cur_epoch = 0, args.epochs
            # Dump2CSV(args, best_metric_list, results_list, [torch.tensor(0)], 'eval.csv')
        
        print_str_classifier = "OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2)
        print(print_str_classifier, flush=True)

        return

    # Training
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    generator.manual_seed(seed)

    # # coder_optimizer = optim.Adam(coder.parameters(), weight_decay=0.)
    # args.transform_func = TransformInput(args)

    # criterion = nn.CrossEntropyLoss().cuda()
    # classifier_optimizer = torch.optim.Adam(list(coder.parameters())+list(classifier.parameters()), 
    #                                         lr=args.learning_rate, weight_decay=args.weight_decay)
    # classifier_optimizer = torch.optim.Adam(list(header.parameters())+list(classifier.parameters()), 
    #                                         lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(classifier_optimizer, step_size=args.epoches//10, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, args.epoches+100)

    start = datetime.now()

    psnr_list = []
    for epoch in range(args.start_epoch, args.epochs):
            
        epoch_start_time = datetime.now()
        # iterate over dataloader
        device = next(coder.parameters()).device

        # TODO: this can be furtehr optimized to accelerate due to fixed coder and Rec data.
        with torch.no_grad():
            coder.eval()
            img_quant_list = [] 
            noised_img_quant_list = [] 
            img_gt_list = []
            for i, sample in enumerate(full_dataloader):
                img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device)
                noise_label = torch.tensor(1, dtype=float).uniform_(0,1)
                noise = norm_idx.clone().uniform_(-0.1/band,0.1/band).clamp_(0,1)
                noised_norm_idx = norm_idx+noise if noise_label >0.5 else norm_idx
                if i > 10 and args.debug:
                    break

                # img_data = torch.autograd.Variable(img_data, requires_grad=True)
                img_data, img_gt, inpaint_mask = args.transform_func(img_data)
                img_out, embed_list, dec_time = coder(img_data, norm_idx)
                img_quant_list.append(img_out.clone())
                # img_gt_list.append(img_gt)

                # img_data = torch.autograd.Variable(img_data, requires_grad=True)
                noised_img_out, noised_embed_list, noised_dec_time = coder(img_data, noised_norm_idx)
                noised_img_quant_list.append(noised_img_out.clone())
                # img_gt_list.append(img_gt)
            
            input_normalize_hat0_ori = torch.cat(img_quant_list, dim=0).type(torch.FloatTensor).to(device)
            input_normalize_hat0_noised = torch.cat(noised_img_quant_list, dim=0).type(torch.FloatTensor).to(device)
            
            # rate = (epoch+1)/(args.epoches)
            # input_normalize_hat1 = torch.where(torch.rand_like(input_normalize_hat0) < 0.5, input_normalize_hat0, input_normalize_hat0 + input_normalize_hat0.uniform_(-0.1,0.1))
            # input_normalize_hat1 = input_gt if (epoch+1)<210 else input_normalize_hat0
            input_normalize_hat1 = input_normalize_hat0_noised if (epoch+1)<210 else input_normalize_hat0_ori
            # input_normalize_hat1 = input_normalize_hat0.detach()
            
            if input_normalize_hat1.size(0) != band:
                raise ValueError('encoder inference error in training between encoder and classifier.')  
            

        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        # torch.set_printoptions(precision=7)
        # att = datetime.now()
        classifier.train()
        header.train()
        
        # 11 is computed by batch and data. For IndianPine, 695/ batchsize(64) = 11;
        for eid in range(11):
            classifier_optimizer.zero_grad()
            
            rec = header(input_normalize_hat1, lambda_rd=beta)
            rec_ori = header(input_normalize_hat0_ori, lambda_rd=beta)
            rec_loss = loss_fn(rec, input_normalize_hat1, args.loss) 

            input_normalize_hat = crop(rec, height, width)
            input_normalize_hat = torch.squeeze(input_normalize_hat, dim=1).permute(1, 2, 0)
            input_normalize_hat_ori = crop(rec_ori, height, width)
            input_normalize_hat_ori = torch.squeeze(input_normalize_hat_ori, dim=1).permute(1, 2, 0)

            x_train_band_hat = train_hat(input_normalize_hat, height, width, band, total_pos_train, patch=args.patches, band_patch=args.band_patches)
            x_train_hat = x_train_band_hat.permute(0,2,1) #[695, 200, 7, 7]
            label_train_loader = samples(x_train_hat, y_train, args.batch_size, shuffle=True, generator=generator)            
            

            batch_data = label_train_loader[eid][0].cuda()
            batch_target = label_train_loader[eid][1].cuda()
            
            batch_pred = classifier(batch_data)
            entropy_loss = criterion(batch_pred, batch_target) 
            loss = 0.4 * entropy_loss + rec_loss # i.e., 1:2.5
            loss.backward()
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            
        # x_test_band_hat = train_hat(input_normalize_hat, height, width, band, total_pos_test, patch=args.patches, band_patch=args.band_patches)
        # x_test_hat = x_test_band_hat.permute(0,2,1) #[695, 200, 7, 7]
        # label_test_loader = samples(x_test_hat, y_test, args.batch_size, shuffle=True, generator=generator)
        
        psnr = compute_psnr(input_normalize_hat1, rec)
        ms_ssim = compute_msssim(input_normalize_hat1, rec)
        print_str = "[{}] Rank:{}, Epoch[{}/{}],  After Header, PSNR: {:.2f} MS_SSIM: {:.4f}".format(
            datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, args.epochs, psnr, ms_ssim)
        print(print_str, flush=True)
        if local_rank in [0, None]:
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            
        scheduler.step()
        train_acc = top1.avg
        train_obj = objs.avg

        
        # OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t) 
        print_str = "[{}] Rank:{}, Epoch[{}/{}],  train_loss: {:.4f} train_acc: {:.4f}".format(
            datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, epoch+1, args.epochs, train_obj, train_acc)
        print(print_str, flush=True)
        if local_rank in [0, None]:
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            pred_psnr = all_reduce([pred_psnr.to(local_rank)])

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5, 7, 10, 15, 20, 25, 35]:
            classifier.eval()
            header.eval()
            
            x_test_band_hat = train_hat(input_normalize_hat_ori, height, width, band, total_pos_test, patch=args.patches, band_patch=args.band_patches)
            x_test_hat = x_test_band_hat.permute(0,2,1) #[695, 200, 7, 7]
            # label_test_loader = samples(x_test_hat, y_test, args.batch_size, shuffle=True, generator=generator)
            Label_test=Data.TensorDataset(x_test_hat,y_test)
            label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)

            
            tar_v, pre_v = valid_epoch(classifier, label_test_loader, criterion, classifier_optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

            results_list, hw = evaluate(coder, full_dataloader, local_rank, args, 
                args.dump_vis if epoch == args.epochs - 1 else False, 
                True if epoch == args.epochs - 1 else False)            
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                print_str = f'Eval at epoch {epoch+1} for {hw}: '
                print_str_classifier = "OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2)
                for i, (metric_name, best_metric_value, metric_value) in enumerate(zip(args.metric_names, best_metric_list, results_list)):
                    best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
                    if 'psnr' in metric_name:
                        writer.add_scalar(f'Val/{metric_name}_{hw}', metric_value.max(), epoch+1)
                        writer.add_scalar(f'Val/best_{metric_name}_{hw}', best_metric_value, epoch+1)
                        writer.add_scalar('Val/OA', OA2, epoch+1)
                        writer.add_scalar('Val/AA', AA_mean2, epoch+1)
                        writer.add_scalar('Val/Kappa', Kappa2, epoch+1)
                        if metric_name == 'pred_seen_psnr':
                            psnr_list.append(metric_value.max())
                        print_str += f'{metric_name}: {RoundTensor(metric_value, 2)} | '
                    best_metric_list[i] = best_metric_value
                print(print_str, flush=True)
                print(print_str_classifier, flush=True)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
                    f.write(print_str_classifier + '\n')
            
            # output classification maps
            pre_u = test_epoch(classifier, label_true_loader, criterion, classifier_optimizer)
            prediction_matrix = np.zeros((height, width), dtype=float)
            for i in range(total_pos_true.shape[0]):
                prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
            plt.subplot(1,1,1)
            plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
            plt.xticks([])
            plt.yticks([])
            # plt.show()
            visual_dir = f'{args.outf}/visualize_model'
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)
            final_mat_path = os.path.join(visual_dir, 'final_mat')
            if not os.path.exists(final_mat_path):
                os.makedirs(final_mat_path)
            final_name_path = os.path.join(final_mat_path,'matrix_'+str(epoch+1)+'.mat')
            savemat(final_mat_path+ '/matrix_'+str(epoch+1)+'.mat',{'P':prediction_matrix, 'label':label})

        # coder_state_dict = coder.state_dict()
        header_state_dict = header.state_dict()
        classifier_state_dict = classifier.state_dict()


        if local_rank in [0, None]:
            # torch.save(save_checkpoint, '{}/coder_latest.pth'.format(args.outf))
            # torch.save(header_state_dict, '{}/header_latest.pth'.format(args.outf))
            # torch.save(classifier_state_dict, '{}/classifier_latest.pth'.format(args.outf))
            if (epoch + 1) % args.epochs == 0:
                args.cur_epoch = epoch + 1
                args.train_time = str(datetime.now() - start)
                # Dump2CSV(args, best_metric_list, results_list, psnr_list, f'epoch{epoch+1}.csv')
                # torch.save(save_checkpoint, f'{args.outf}/coder_epoch{epoch+1}.pth')
                torch.save(header_state_dict, f'{args.outf}/header_epoch{epoch+1}.pth')
                torch.save(classifier_state_dict, f'{args.outf}/classifier_epoch{epoch+1}.pth')
                if best_metric_list[0]==results_list[0]:
                    # torch.save(save_checkpoint, f'{args.outf}/coder_best.pth')
                    torch.save(header_state_dict, f'{args.outf}/header_best.pth')
                    torch.save(classifier_state_dict, f'{args.outf}/classifier_best.pth')

    if local_rank in [0, None]:
        print(f"Training complete in: {str(datetime.now() - start)}")
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            print_str_time = "Training complete in:"+ str(datetime.now() - start)
            f.write(print_str_time + '\n')


@torch.no_grad()
def evaluate(model, full_dataloader, local_rank, args, 
    dump_vis=False, huffman_coding=False):
    img_embed_list = []
    model_list, quant_ckt = quant_model(model, args)
    metric_list = [[] for _ in range(len(args.metric_names))]
    for model_ind, cur_model in enumerate(model_list):
        time_list = []
        img_quant_list = [] 
        cur_model.eval()
        device = next(cur_model.parameters()).device
        if dump_vis:
            visual_dir = f'{args.outf}/visualize_model' + ('_quant' if model_ind else '_orig')
            print(f'Saving predictions to {visual_dir}...')
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)        

        for i, sample in enumerate(full_dataloader):
            img_data, norm_idx, img_idx = data_to_gpu(sample['img'], device), data_to_gpu(sample['norm_idx'], device), data_to_gpu(sample['idx'], device)
            if i > 10 and args.debug:
                break
            img_data, img_gt, inpaint_mask = args.transform_func(img_data)
            img_out, embed_list, dec_time = cur_model(img_data, norm_idx,  input_embed = 1 if model_ind else None)
            if model_ind == 0:
                img_embed_list.append(embed_list[0])
            
            # collect decoding fps
            time_list.append(dec_time)
            if args.eval_fps:
                time_list.pop()
                for _ in range(100):
                    img_out, embed_list, dec_time = cur_model(img_data, norm_idx, embed_list[0])
                    time_list.append(dec_time)
            
            img_quant_list.append(img_out.clone())

            # x_test_band_hat = train_hat(img_quant_list, height, width, band, total_pos_test, patch=args.patches, band_patch=args.band_patches)
            # x_test_hat = x_test_band_hat.permute(0,2,1) #[695, 200, 7, 7]
            # # label_test_loader = samples(x_test_hat, y_test, args.batch_size, shuffle=True, generator=generator)
            # Label_test=Data.TensorDataset(x_test_hat,y_test)
            # label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)

            # compute psnr and ms-ssim
            pred_psnr, pred_ssim = psnr_fn_batch([img_out], img_gt), msssim_fn_batch([img_out], img_gt)
            for metric_idx, cur_v in  enumerate([pred_psnr, pred_ssim]):
                for batch_i, cur_img_idx in enumerate(img_idx):
                    metric_idx_start = 2 if cur_img_idx in args.val_ind_list else 0
                    metric_list[metric_idx_start+metric_idx+4*model_ind].append(cur_v[:,batch_i])

            # dump predictions
            if dump_vis:
                for batch_ind, cur_img_idx in enumerate(img_idx):
                    full_ind = i * args.batchSize + batch_ind
                    dump_img_list = [img_data[batch_ind], img_out[batch_ind]]
                    temp_psnr_list = ','.join([str(round(x[batch_ind].item(), 2)) for x in pred_psnr])
                    concat_img = torch.cat(dump_img_list, dim=2)    #img_out[batch_ind], 
                    save_image(concat_img, f'{visual_dir}/pred_{full_ind:04d}_{temp_psnr_list}.png')

            # print eval results and add to log txt
            if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
                avg_time = sum(time_list) / len(time_list)
                fps = args.batchSize / avg_time
                print_str = '[{}] Rank:{}, Eval at Step [{}/{}] , FPS {}, '.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, i+1, len(full_dataloader), round(fps, 1))
                metric_name = ('quant' if model_ind else 'pred') + '_seen_psnr'
                for v_name, v_list in zip(args.metric_names, metric_list):
                    if metric_name in v_name:
                        cur_value = torch.stack(v_list, dim=-1).mean(-1) if len(v_list) else torch.zeros(1)
                        print_str += f'{v_name}: {RoundTensor(cur_value, 2)} | '
                if local_rank in [0, None]:
                    print(print_str, flush=True)
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')
    
        input_normalize_hat = torch.cat(img_quant_list, dim=0).type(torch.FloatTensor).to(device)
        
        # embedding quantization
        if model_ind == 0:
            vid_embed = torch.cat(img_embed_list, 0) 
            quant_embed, _ = quant_tensor(vid_embed, args.quant_embed_bit)

        # Collect results from 
        results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in metric_list]
        args.fps = fps
        h,w = img_data.shape[-2:]
        cur_model.train()
        if args.distributed and args.ngpus_per_node > 1:
            for cur_v in results_list:
                cur_v = all_reduce([cur_v.to(local_rank)])

        # Dump predictions and concat into videos
        if dump_vis and args.dump_videos:
            gif_file = os.path.join(args.outf, 'gt_pred' + ('_quant.gif' if model_ind else '.gif'))
            with imageio.get_writer(gif_file, mode='I') as writer:
                for filename in sorted(os.listdir(visual_dir)):
                    image = imageio.v2.imread(os.path.join(visual_dir, filename))
                    writer.append_data(image)
            if not args.dump_images:
                shutil.rmtree(visual_dir)
            # optimize(gif_file)
        
    # dump quantized checkpoint, and decoder
    if local_rank in [0, None] and quant_ckt != None:
        quant_vid = {'embed': quant_embed, 'model': quant_ckt}
        torch.save(quant_vid, f'{args.outf}/quant_vid.pth')
        torch.jit.save(torch.jit.trace(HINERDecoder(model), (vid_embed[:2])), f'{args.outf}/img_decoder.pth')
        # huffman coding
        if huffman_coding:
            quant_v_list = quant_embed['quant'].flatten().tolist()
            tmin_scale_len = quant_embed['min'].nelement() + quant_embed['scale'].nelement()
            for k, layer_wt in quant_ckt.items():
                quant_v_list.extend(layer_wt['quant'].flatten().tolist())
                tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

            # get the element name and its frequency
            unique, counts = np.unique(quant_v_list, return_counts=True)
            num_freq = dict(zip(unique, counts))

            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(quant_v_list)
            sym_bit_dict = {}
            for k, v in codec.get_code_table().items():
                sym_bit_dict[k] = v[0]

            # total bits for quantized embed + model weights
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            args.bits_per_param = total_bits / len(quant_v_list)
            
            # including the overhead for min and scale storage, 
            total_bits += tmin_scale_len * 16               #(16bits for float16)
            args.full_bits_per_param = total_bits / len(quant_v_list)

            # bits per pixel
            args.total_bpp = total_bits / args.final_size / args.full_data_length
            print(f'After quantization and encoding: \n bits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}')
    # import pdb; pdb.set_trace; from IPython import embed; embed()     

    return results_list, (h,w)


def main(argv):
    args = parse_args(argv)
    seed_all(args.seed)

    torch.set_printoptions(precision=4)
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    # model setting and hyper parameters
    if 'hiner' in args.arch:
        args.enc_strds = args.dec_strds
    
    args.enc_strd_str, args.dec_strd_str = ','.join([str(x) for x in args.enc_strds]), ','.join([str(x) for x in args.dec_strds])
    extra_str = 'Size{}_ENC_{}_{}_DEC_{}_{}_{}{}{}'.format(args.modelsize, args.conv_type[0], args.enc_strd_str, 
        args.conv_type[1], args.dec_strd_str, '' if args.norm == 'none' else f'_{args.norm}', 
        '_dist' if args.distributed else '', '_shuffle_data' if args.shuffle_data else '',)
    args.quant_str = f'quant_M{args.quant_model_bit}_E{args.quant_embed_bit}'
    embed_str = f'{args.embed}_Dim{args.enc_dim}'
    exp_id = f'{args.vid}/{args.data_split}_{embed_str}_FC{args.fc_hw}_KS{args.ks}_RED{args.reduce}_low{args.lower_width}_blk{args.num_blks}' + \
            f'_e{args.epochs}_b{args.batchSize}_{args.quant_str}_lr{args.lr}_{args.lr_type}_{args.loss}_{extra_str}{args.act}{args.block_params}{args.suffix}_decay{args.weight_decay}'
    args.exp_id = exp_id

    # output dir
    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)



if __name__ == '__main__':
    main(sys.argv[1:])


