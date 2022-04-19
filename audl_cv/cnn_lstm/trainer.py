import os
from tqdm.auto import tqdm
import logging
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 


class Trainer:
    def __init__(self, model, train_dataset, dev_dataset=None):
        self.model = model
        if not dev_dataset:
            dev_dataset = train_dataset
            
        self.set_seed()
        self._get_dataloader(train_dataset, dev_dataset)
        self._get_optimizer()

    def set_seed(self):
        self.seed = 123
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _get_dataloader(self, train_dataset, dev_dataset):
        self.train_loader = DataLoader(train_dataset,
                                batch_size=2,
                                collate_fn=train_dataset.collate_fn,
                                shuffle=True,
                                drop_last=True
                                )
        self.dev_loader = DataLoader(dev_dataset,
                                batch_size=2,
                                collate_fn=dev_dataset.collate_fn,
                                shuffle=False,
                                drop_last=False
                                )
            
    def _get_optimizer(self):
        model_params = list(self.model.named_parameters())
        no_decay = ['bias']
        optimized_params = [
            {
                'params':[p for n, p in model_params if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }   
        ]
        
        self.optimizer = AdamW(optimized_params, lr=0.001)
    
    def run_train(self, run_name, n_epochs):
        best_val_loss = self.run_validation()
        
        for epoch in range(n_epochs):
            pbar = tqdm(self.train_loader)
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            
            for i, batch in enumerate(pbar):
                inputs = batch[0].to(self.model.device())
                labels = batch[1].to(self.model.device())
                batch = (inputs, labels)
                
                batch_loss = self._training_step(batch)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad(set_to_none=True)
                
                pbar.set_description(f'(Training) Epoch: {epoch} - Loss: {batch_loss:.4f}')
                
            val_loss = self.run_validation()
            
            if val_loss < best_val_loss:
                logger.info(f'New best validatoin loss at {val_loss:.4f}, saving checkpoint')
                best_val_loss = val_loss
                ckpt_path = os.path.join(run_name + '.pt')
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f'New checkpoint saved at {ckpt_path}')
                
            if (val_loss >= best_val_loss) and self.model.frozen == True:
                self.model.frozen = False
                for g in self.optimizer.param_groups:
                    g['lr'] = 0.0003
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                    
    def run_validation(self):
        pbar = tqdm(enumerate(self.dev_loader), total=len(self.dev_loader))
        self.model.eval()
        epoch_loss = 0
        
        for i, batch in pbar:
            loss = self._prediction_step(batch)
            pbar.set_description(f'(Validating) Loss: {loss:.4f}')
            epoch_loss += loss
        
        logger.info(f' Validation loss: {epoch_loss:.4f}')
        
        return epoch_loss
    
    def _training_step(self, batch):
        loss = self.model(batch)
        loss.backward()
        
        return loss.detach()
    
    @torch.no_grad()
    def _prediction_step(self, batch):
        loss = self.model(batch)
        
        return loss.detach()
    
if __name__ == '__main__':
    import os
    
    import torchvision.models as models
    from torch.utils.data import DataLoader
    
    from audl_cv.cnn_lstm.model import CNNLSTM
    from audl_cv.cnn_lstm.dataset import AUDLDataset
    
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