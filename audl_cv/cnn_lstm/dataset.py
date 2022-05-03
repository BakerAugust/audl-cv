from tqdm.auto import tqdm
import pandas as pd
import cv2
import os
import numpy as np
import glob

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class AUDLDataset(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.clip_length = 80
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        paths = self._get_paths(data_path)
        # paths = [data_path]
        self.clips, self.labels = self._get_data(paths)
        self.n_samples = len(self.clips)
        
    def _get_paths(self, data_path):
        annotation_path = data_path.split('\\') + ['possession_annotations', '*.feather']
        annotation_path = os.path.join(*annotation_path)
        
        annotations = []
        
        for file in glob.glob(annotation_path):
            annotations.append(file)
            
        return annotations
    
    def _get_data(self, paths):
        clips = []
        labels = []
        
        for path in tqdm(paths):
            path = os.path.join(*path.split('\\'))
            # annotation_path = path + '.feather'
            clip_path = path.replace('annotations', 'clips').replace('.feather', '.mp4')
            
            annotation = pd.read_feather(path)
            label = annotation[['x', 'y']].to_numpy(dtype=np.float32)
            
            frames = set(annotation['frame_number'].tolist())
            video = cv2.VideoCapture(clip_path)
            success, image = video.read()
            frame_out = []
            i_frame = 0
            
            while success:
                if i_frame in frames:
                    image = cv2.resize(image, (128, 128))
                    frame_out.append(self.preprocess(image))
                i_frame += 1
                success, image = video.read()
            
            frame_out = np.stack(frame_out, axis=0) 
            
            frame_out_samples = self._split_to_samples(frame_out)
            label_samples = self._split_to_samples(label)
        
            clips.extend(frame_out_samples)
            labels.extend(label_samples)
    
        return clips, labels
    
    def _split_to_samples(self, array):
        samples = np.split(array, np.arange(self.clip_length, array.shape[0], self.clip_length), axis=0)
        samples = [torch.tensor(sample) for sample in samples]
        
        return samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, item):
        clip = self.clips[item]
        # clip = self.preprocess(clip)
        label = self.labels[item]
        
        return clip, label
    
    def collate_fn(self, batch):
        clips, labels = zip(*batch)
        
        clips = pad_sequence(clips, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)
        
        return clips, labels
            
if __name__ == '__main__':
    print('hello world')
    from torch.utils.data import DataLoader
    
    path = 'data\\val'
    
    dataset = AUDLDataset(None, path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    print(dataset.n_samples)
    for i, sample in enumerate(dataloader):
        if i == 10: break
        print(sample[0].shape)
        print(sample[0].dtype)
        print(sample[1].shape)
        print(sample[1].dtype)
        print('====')

    
    # clip_path = path.replace('annotations', 'clips') + '.mp4'    
    # video = cv2.VideoCapture(clip_path)
    # success, image = video.read()
    # print(image/255.)