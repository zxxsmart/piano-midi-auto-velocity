import os
import torch.utils.data as data
import numpy as np
import random
import torch
from mido import MidiFile

class MidiDataset(data.Dataset):
    """
    dataloader for midi notes and velocities
    """
    def __init__(self, mid_root, trainsize):
        self.mids = [mid_root + f for f in os.listdir(mid_root) if f.endswith('.mid')]
        self.mids = sorted(self.mids)
        self.size = len(self.mids)
        self.trainsize = trainsize

    def __getitem__(self, index):
        mid = self.mid_loader(self.mids[index])
        notes, gt = self.get_notes_velocity(mid)
        cut = random.randint(0, notes.shape[1]-self.trainsize-1)
        return notes[:,cut:cut+self.trainsize], gt[:,cut:cut+self.trainsize]

    def mid_loader(self, path):
        mid = MidiFile(path)
        return mid

    def get_notes_velocity(self, mid):
        track = mid.tracks[1]
        times = 0
        i = 0

        events = {}

        t0 = track[0].time

        for msg in track:
            times += msg.time

        times -= t0

        notes_data = np.zeros((88,times)).astype('float') #notes: (88, midi_length) =1 when note is on
        velocity_data = np.zeros((88,times)).astype('float') #velocity: (88, midi_length)

        previous_ticks = -t0
        for msg in track:
            this_ticks = msg.time + previous_ticks
            if this_ticks >= 0:
                # diff_ticks = msg.time
                if msg.type=='note_on':
                    if msg.note in events:
                        info = events[msg.note]
                        notes_data[msg.note-21,info[0]:this_ticks] = 1
                        velocity_data[msg.note-21,info[0]:this_ticks] = info[1]
                        events.pop(msg.note)
                    else:
                        events[msg.note] = [this_ticks,msg.velocity]           
                previous_ticks = this_ticks

        return notes_data, ((velocity_data/127)-0.5)*2
    
    def resize(self, input):
        pass

    def __len__(self):
        return self.size


def get_loader(mid_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = MidiDataset(mid_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, mid_root, trainsize):
        self.mids = [mid_root + f for f in os.listdir(mid_root) if f.endswith('.mid')]
        self.mids = sorted(self.mids)
        self.size = len(self.mids)
        self.trainsize = trainsize
        self.index = 0

    def load_data(self):
        mid = self.mid_loader(self.mids[self.index])
        notes = self.get_notes_velocity(mid)
        return notes

    def mid_loader(self, path):
        mid = MidiFile(path)
        return mid

    def get_notes_velocity(self, mid):
        track = mid.tracks[1]
        times = 0
        i = 0

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
                if msg.type=='note_on':
                    if msg.note in events:
                        info = events[msg.note]
                        notes_data[msg.note-21,info[0]:this_ticks] = 1
                        events.pop(msg.note)
                    else:
                        events[msg.note] = [this_ticks,msg.velocity]           
                previous_ticks = this_ticks

        return notes_data



if __name__ == '__main__':
    train_loader = get_loader('midis/', batchsize=2, trainsize= 10000)

    for i, pack in enumerate(train_loader, start=1):
        
        pass