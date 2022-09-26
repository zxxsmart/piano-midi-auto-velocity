import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime

from model import *
from dataloader import get_loader
from utils import clip_gradient, adjust_lr, AvgMeter

# import torch.nn.functional as F
# import numpy as np
# from tqdm import tqdm
# from torchstat import stat

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(train_loader, model, optimizer, epoch, loss_func):
    model.train()
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        mids, gts = pack
        mids = Variable(mids.type(torch.FloatTensor)).cuda()
        gts = Variable(gts.type(torch.FloatTensor)).cuda()
        trainsize = opt.trainsize

        out = model(mids)

        loss_last = loss_func(out, gts)
        
        loss = loss_last * 10
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        loss_record.update(loss.data, opt.batchsize)

        # ---- train visualization ----
        if i % 30 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    ' loss: {:0.4f}]'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            loss_record.show()))
                            
    save_path = 'models_save/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    
    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), save_path + str(loss)+'.pth' )
        print('[Saving Snapshot:]', loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=1e-3, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=4096, help='training dataset size')
    
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    
    parser.add_argument('--decay_epoch', type=int,
                        default=100, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str,
                        default='midis/', help='path to train dataset')
    
    parser.add_argument('--test_path', type=str,
                        default='./' , help='path to testing')
    
    parser.add_argument('--train_save', type=str,
                        default='model_save')
    
    opt = parser.parse_args()

    # ---- build models ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet1D(88,88).to(device)
    # model = GruRNN(88,88,10).cuda()

    # Multi GPU
    # model = nn.DataParallel(model)
    # model = model.cuda()

    loss_func = torch.nn.MSELoss()

    params = model.parameters()

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)
        
    print(optimizer)

    mid_root = opt.train_path

    train_loader = get_loader(mid_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=2)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, loss_func)
