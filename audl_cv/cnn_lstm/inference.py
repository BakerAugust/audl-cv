import os
from sklearn import model_selection
from tqdm import tqdm
import json
import logging
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Inference:
    def __init__(self, model, dataset, ckpt_path):
        self.model = model
        self.dataset = dataset
        
        self._set_seed()
        self._get_dataloader()
        self._init_model(ckpt_path)
        
    def _set_seed(self):
        self.seed = 123
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _get_dataloader(self):
        self.loader = DataLoader(self.dataset, 
                                batch_size=4,
                                collate_fn = self.dataset.collate_fn,
                                shuffle=False)
    
    def _init_model(self, path):
        path = os.path.join(*path.split('\\'))
        self.model.load_state_dict(torch.load(path))
        logger.info(f' Loaded model checkpoint from {path}')
    
    @torch.no_grad()
    def run_test(self):
        pbar = tqdm(self.loader)
        self.model.eval()
        preds, golds = [], []
        total_loss = 0
        out = []
        
        for i, batch in enumerate(pbar):
            inputs = batch[0].to(self.model.device())
            labels = batch[1].to(self.model.device())
            batch = (inputs, labels)
            
            pred = self.model(batch, return_type='preds')
            out.append({
                'preds': pred.cpu().numpy().tolist(),
                'labels': labels.cpu().numpy().tolist()
                })
            # loss = torch.sqrt(F.mse_loss(pred, labels))
            loss = F.l1_loss(pred, labels)
            total_loss += loss
        
        test_loss = total_loss / len(self.loader)        
        logger.info(f' Test set MAE: {test_loss:4f}')
        
        # with open('predpred.json', 'w') as f:
        #     json.dump(out, f)
        
if __name__ == '__main__':
    print('hello world')    
    import torchvision.models as models
    
    from audl_cv.cnn_lstm.model import CNNLSTM
    from audl_cv.cnn_lstm.dataset import AUDLDataset
    
    data_path = 'data\\val'
    ckpt_path = 'hw128_seqlen600_mnasnet10_hidden512_lstm2.pt'
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    pretrained_model = models.mnasnet1_0(pretrained=True)
    encoder = torch.nn.Sequential(*(list(pretrained_model.children())[:-1]))
    params = {
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.25
    }
    
    model = CNNLSTM(encoder, params)
    dataset = AUDLDataset(None, data_path)
    
    inference = Inference(model, dataset, ckpt_path)
    inference.run_test()