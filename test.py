import torch
import torch.nn.functional as F
import numpy as np
from mido import MidiFile
from tqdm import tqdm
import os, argparse

from model import *
# from dataloader import get_loader

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=4096, help='testing size')
parser.add_argument('--pth_path', type=str, default="models_save/model_save/tensor(0.2475, device='cuda:0', grad_fn=<MulBackward0>).pth")
parser.add_argument('--mid_path', type=str, default='test_mids/', help='input path')
parser.add_argument('--out_path', type=str, default='output/', help='output path')
parser.add_argument('--replace_zero', type=int, default=60, help='replace 0 velocity with 1-127')
opt = parser.parse_args()

model = Unet1D(88,88)
# model = GruRNN(88,88,10)
model.load_state_dict(torch.load(opt.pth_path))
model.to(device)
model.eval()

os.makedirs(opt.out_path, exist_ok=True)

for mid_name in tqdm(os.listdir(opt.mid_path)):
    mid = MidiFile(opt.mid_path+mid_name)
    ind = 0
    track = mid.tracks[0]
    for i, t in enumerate(mid.tracks):
        if len(t)>len(track):
            track = t
            ind = i
    times = 0
    events = {}
    t0 = track[0].time
    for msg in track:
        times += msg.time
    times -= t0
    notes_data = np.zeros((88,times)).astype('float')
    previous_ticks = -t0
    for msg in track:
        this_ticks = msg.time + previous_ticks
        if this_ticks >= 0:
            # diff_ticks = msg.time
            if msg.type=='note_on' and msg.velocity!=0:
                events[msg.note] = [this_ticks,msg.velocity]
            elif (msg.type=='note_on' and msg.velocity==0) or msg.type=='note_off':
                if msg.note in events:
                    info = events[msg.note]
                    if msg.note-21<0 or msg.note-21>=88:
                        continue
                    notes_data[msg.note-21,info[0]:this_ticks] = 1
                    events.pop(msg.note)
                # else:
                #     print(msg)
            previous_ticks = this_ticks

    velocity_data = np.zeros((88,times)).astype('int')

    i = 0    
    while(i<=times):
        if i+opt.testsize>=times:
            piece = notes_data[:,i:]
            piece = np.pad(piece, ((0,0),(0,opt.testsize-piece.shape[1])))
        else:
            piece = notes_data[:,i:i+opt.testsize]
        piece = piece.reshape((1,88,opt.testsize))
        if piece.max()==0:
            print(piece)
        piece = torch.Tensor(piece).to(device)
        out = model(piece)
        out = ((out/2+0.5)*127).cpu().detach().numpy()
        # if out==0:
        #     pass
        out = np.round(out.reshape((88,opt.testsize)))

        if i+opt.testsize>=times:
            velocity_data[:,i:] = out[:,:times-i]
            break
        else:
            velocity_data[:,i:i+opt.testsize] = out
        
        i = i+opt.testsize
        
    events = {}
    previous_ticks = -t0
    for i, msg in enumerate(track):
        this_ticks = msg.time + previous_ticks
        if this_ticks >= 0:
            if msg.type=='note_on' and msg.velocity!=0:
                events[msg.note] = [i, this_ticks]
            elif (msg.type=='note_on' and msg.velocity==0) or msg.type=='note_off':
                if msg.note-21<0 or msg.note-21>=88:
                    continue
                [a, start] = events[msg.note]
                end = this_ticks
                if start==end:
                    continue
                # print(i)
                v = int(np.round(np.mean(velocity_data[msg.note-21,start:end])))
                if v>0 and v<=127:
                    track[a].velocity = int(np.round(np.mean(velocity_data[msg.note-21,start:end])))
                else:
                    track[a].velocity = opt.replace_zero#track[a-1].velocity
                    print(msg, v)       
            previous_ticks = this_ticks

    mid.tracks[ind] = track

    mid.save(opt.out_path+mid_name[:-4]+'_out.mid')