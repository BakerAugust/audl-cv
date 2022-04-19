import os

import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from audl_cv.cnn_lstm.model import CNNLSTM
from audl_cv.cnn_lstm.dataset import AUDLDataset
from audl_cv.cnn_lstm.trainer import Trainer

if __name__ == '__main__':
    
    pretrained_model = models.mnasnet1_0(pretrained=True)
    encoder = torch.nn.Sequential(*(list(pretrained_model.children())[:-1]))
    params = {
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.25
    }
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    data_path = 'data\possession_annotations\\2021-08-28-DAL-SD-1-672-693'
    
    
    model = CNNLSTM(encoder, params)
    dataset = AUDLDataset(None, data_path)
    trainer = Trainer(model, dataset)
    
    trainer.run_train('initial_run', 5)