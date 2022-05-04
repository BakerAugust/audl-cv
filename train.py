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
    train_path = 'data\\train'
    val_path = 'data\\val'
    
    
    model = CNNLSTM(encoder, params)
    train_dataset = AUDLDataset(None, train_path)
    val_dataset = AUDLDataset(None, val_path)
    trainer = Trainer(model, train_dataset, val_dataset)
    
    trainer.run_train('initial_run', 5)