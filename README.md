# piano-midi-auto-velocity
Automatically adjust the velocity of piano midi notes to make midi sound more like a real performance.

## Model and dataset
Used a simple UNet(1D) to regress the velocity of piano midi notes. Training with the [GiantMIDI-Piano](https://github.com/joann8512/GiantMIDI-Piano) dataset. Input is a slice of midi notes with only pitch and time information, and output is the velocity of each notes.

## Requirements
Install PyTorch following https://pytorch.org/. Install mido.

## Test
There is a pretrained model in the release. Put input midis in `test_mids/` and run: 
```
python test.py --pth_path=Model Path
```
the output midis will save in `out_put/`. For the input midi file, all notes should put in a single track with pitch in the range of normal 88k piano(A0-C8).

## Train
If you want to train, please put the midi of the dataset into `midis/` and run train.py. Currently there are problems with the way of loading data, resulting in low GPU utilization. 

## Result
I don't know if the current results are reasonable, is like a randomization based on pitch.