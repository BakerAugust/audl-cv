import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, pretrained_encoder, params):
        super(CNNLSTM, self).__init__()
        self.encoder = pretrained_encoder
        self.lstm = nn.LSTM(input_size=40960,
                            hidden_size=params['hidden_size'],
                            num_layers=params['num_layers'],
                            batch_first=True,
                            dropout=params['dropout'])        
        self.linear_out = nn.Linear(params['hidden_size'], 2)
        
        self.frozen = True
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def device(self):
        return next(self.parameters()).device
    
    def loss_fn(self, preds, labels):       
        return F.mse_loss(preds, labels)
    
    def forward(self, batch, return_type='loss'):
        if return_type == 'loss':
            bs, seq_len, h, w, c = batch[0].size()
            reshaped_img = torch.reshape(batch[0], (bs*seq_len, h, w, c))
            # print('1', reshaped_img.size())
            reshaped_img = torch.permute(reshaped_img, (0, 3, 1, 2))
            # print('2', reshaped_img.size())
            img_embedding = self.encoder(reshaped_img)
            # print('3', img_embedding.size())
            _, emb_1, emb_2, emb_3 = img_embedding.size()
            img_embedding = torch.reshape(img_embedding, (bs, seq_len, emb_1, emb_2, emb_3))
            # print('4', img_embedding.size())
            img_embedding = torch.flatten(img_embedding, start_dim=2)
            # print('5', img_embedding.size())
            lstm_out, _ = self.lstm(img_embedding)
            # print('6', lstm_out.size())
            preds = self.linear_out(lstm_out)
            # print(preds.size())

            return self.loss_fn(preds, batch[1])
        
if __name__ == '__main__':
    encoder = models.mnasnet1_0(pretrained=True)
    new_encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
    params = {
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.25
    }
    
    device = 'cpu'
    
    model = CNNLSTM(new_encoder, params)
    batch = (torch.ones(4, 200, 120, 240, 3, device=device), 
             torch.ones(4, 200, 2, device=device))
    model.to(device)
    
    loss = model(batch)
    print(loss)