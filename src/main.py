# -*- coding: utf-8 -*-

import os
import tqdm
import argparse
import random
import time
import numpy as np

from model import STMT
from dataset import NTU_RGBD_norm_samplenum as NTU_RGBD
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def copy_parameters(model, pretrained, verbose=True):

    model_dict = model.state_dict()
    pretrained_dict = pretrained['state_dict']
    pretrained_dict_rename = {}
    for k, v in pretrained_dict.items():
        pretrained_dict_rename[k.replace('feat','module')] = v
    pretrained_dict = pretrained_dict_rename
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def main_woker(opt):
    os.makedirs(opt.save_root_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        filename=os.path.join(opt.save_root_dir,
                                              'train00.log'),
                        level=logging.INFO)

    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    try:
        os.makedirs(opt.save_root_dir)
    except OSError:
        pass

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    data_train = NTU_RGBD(root_path=opt.root_path, opt=opt,
                          DATA_CROSS_VIEW=False,
                          full_train=True,
                          validation=False,
                          test=False,
                          Transform=True
                          )
    train_loader = DataLoader(dataset=data_train, batch_size=32,
                              shuffle=True, drop_last=True, num_workers=8)
    data_val = NTU_RGBD(root_path=opt.root_path, opt=opt,
                        DATA_CROSS_VIEW=False,
                        full_train=False,
                        validation=True,
                        test=False,
                        Transform=False
                        )
    val_loader = DataLoader(dataset=data_val, batch_size=32, num_workers=8)

    netR = STMT(opt)
    netR = torch.nn.DataParallel(netR).cuda()
    netR.cuda()
    print(netR)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(netR.parameters(), lr=0.0001
    , weight_decay = 1e-4)

    scheduler = CosineAnnealingLR(optimizer, 150, eta_min=0.0001)
    load_path = './ckpt/mvm_pretrained.pth'
    checkpoint = torch.load(load_path)
    copy_parameters(netR, checkpoint, verbose=True)
    print('load ', load_path)
    for epoch in range(opt.nepoch):
        if True:
            # evaluate mode
            torch.cuda.synchronize()
            netR.eval()
            conf_mat = np.zeros([opt.Num_Class, opt.Num_Class])
            # conf_mat60 = np.zeros([20, 20])
            acc = 0.0
            loss_sigma = 0.0

            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    torch.cuda.synchronize()

                    points4DV_T, geodist, label, v_name = data
                    points4DV_T, label = points4DV_T.cuda(), label.cuda()
                    prediction = netR(points4DV_T,geodist)

                    loss = criterion(prediction, label)
                    _, predicted = torch.max(prediction.data, 1)
                    loss_sigma += loss.item()

                    for j in range(len(label)):
                        cate_i = label[j].cpu().numpy()
                        pre_i = predicted[j].cpu().numpy()
                        conf_mat[cate_i, pre_i] += 1.0

            print(
                'KIT:{:.2%} -correct number {}--all number {}===Average loss:{:.6%}'.format(
                    conf_mat.trace() / conf_mat.sum(),
                    conf_mat.trace(),
                    conf_mat.sum(), loss_sigma / (i + 1) / 2))
            logging.info(
                '#################{} --epoch{} set Accuracy:{:.2%}--correct number {}--all number {}===Average loss:{}'.format(
                    'Valid', epoch, conf_mat.trace() / conf_mat.sum(),
                    conf_mat.trace(), conf_mat.sum(),
                                    loss_sigma / (i + 1)))
        if epoch != 0:
            scheduler.step(epoch)

        # switch to train mode
        torch.cuda.synchronize()
        netR.train()
        acc = 0.0
        loss_sigma = 0.0
        total1 = 0.0
        timer = time.time()

        for i, data in enumerate(tqdm(train_loader, 0)):
            torch.cuda.synchronize()
            points4DV_T, geodist, label, v_name = data
            points4DV_T, label = points4DV_T.cuda(), label.cuda()
            prediction = netR(points4DV_T,geodist)
            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            loss_sigma += loss.item()
            _, predicted = torch.max(prediction.data, 1)
            acc += (predicted == label).cpu().sum().numpy()
            total1 += label.size(0)

        acc_avg = acc / total1
        loss_avg = loss_sigma / total1
        print(
            '======>>>>> Online epoch: #%d, lr=%.10f,Acc=%f,correctnum=%f,allnum=%f,avg_loss=%f  <<<<<======' % (
                epoch, scheduler.get_lr()[0], acc_avg, acc, total1, loss_avg))
        print("Epoch: " + str(epoch) + " Iter: " + str(i) + " Acc: " + (
                "%.2f" % acc_avg) + " Classification Loss: " + str(
            loss_avg))
        logging.info(
            '======>>>>> Online epoch: #%d, lr=%.10f,Acc=%f,correctnum=%f,allnum=%f,avg_loss=%f  <<<<<======' % (
                epoch, scheduler.get_lr()[0], acc_avg, acc, total1, loss_avg))
        logging.info("Epoch: " + str(epoch) + " Iter: " + str(i) + " Acc: " + (
                "%.2f" % acc_avg) + " Classification Loss: " + str(
            loss_avg))

        ave_states = {
            'epoch': epoch,
            'state_dict': netR.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(ave_states,
                   '%s/stmt_%d.pth' % (opt.save_root_dir, epoch))

def main(args=None):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument('--batchSize', type=int, default=32,
                        help='input batch size')  # ￥￥￥￥
    parser.add_argument('--nepoch', type=int, default=150,
                        help='number of epochs to train for')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=3,
                        help='number of input point features')
    parser.add_argument('--temperal_num', type=int, default=3,
                        help='number of input point features')
    parser.add_argument('--pooling', type=str, default='concatenation',
                        help='how to aggregate temporal split features: vlad | concatenation | bilinear')
    parser.add_argument('--dataset', type=str, default='ntu60',
                        help='how to aggregate temporal split features: ntu120 | ntu60')

    parser.add_argument('--weight_decay', type=float, default=0.0008,
                        help='weight decay (SGD only)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate at t=0')  # ￥￥￥￥
    parser.add_argument('--gamma', type=float, default=0.5, help='')  # ￥￥￥￥
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (SGD only)')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')

    parser.add_argument('--root_path', type=str,
                        default='',
                        help='preprocess folder')

    parser.add_argument('--save_root_dir', type=str,
                        default='',
                        help='output folder')

    parser.add_argument('--model', type=str, default='',
                        help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default='',
                        help='optimizer name for training resume')

    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    parser.add_argument('--main_gpu', type=int, default=0,
                        help='main GPU id')  # CUDA_VISIBLE_DEVICES=0 python train.py

    ########
    parser.add_argument('--Seg_size', type=int, default=1,
                        help='number of frame in seg')
    parser.add_argument('--stride', type=int, default=1, help='stride of seg')
    parser.add_argument('--all_framenum', type=int, default=24,
                        help='number of action frame')
    parser.add_argument('--framenum', type=int, default=24,
                        help='number of action frame')
    parser.add_argument('--EACH_FRAME_SAMPLE_NUM', type=int, default=512,
                        help='number of sample points in each frame')
    parser.add_argument('--T_knn_K', type=int, default=48,
                        help='K for knn search of temperal stream')
    parser.add_argument('--T_knn_K2', type=int, default=16,
                        help='K for knn search of temperal stream')
    parser.add_argument('--T_sample_num_level1', type=int, default=128,
                        help='number of first layer groups')
    parser.add_argument('--T_sample_num_level2', type=int, default=32,
                        help='number of first layer groups')
    parser.add_argument('--T_ball_radius', type=float, default=0.2,
                        help='square of radius for ball query of temperal stream')

    parser.add_argument('--learning_rate_decay', type=float, default=1e-7,
                        help='learning rate decay')

    parser.add_argument('--size', type=str, default='full',
                        help='how many samples do we load: small | full')
    parser.add_argument('--SAMPLE_NUM', type=int, default=2048,
                        help='number of sample points')

    parser.add_argument('--Num_Class', type=int, default=56,
                        help='number of outputs')
    parser.add_argument('--knn_K', type=int, default=64,
                        help='K for knn search')
    parser.add_argument('--sample_num_level1', type=int, default=512,
                        help='number of first layer groups')
    parser.add_argument('--sample_num_level2', type=int, default=128,
                        help='number of second layer groups')
    parser.add_argument('--ball_radius', type=float, default=0.1,
                        help='square of radius for ball query in level 1')
    parser.add_argument('--ball_radius2', type=float, default=0.2,
                        help='square of radius for ball query in level 2')

    opt = parser.parse_args()
    print(opt)
    main_woker(opt)



if __name__ == '__main__':
    main()

