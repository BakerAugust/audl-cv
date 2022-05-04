import logging

logging.basicConfig
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_test():
    import torch
    import torchvision.models as models
    
    from audl_cv.cnn_lstm.model import CNNLSTM
    from audl_cv.cnn_lstm.dataset import AUDLDataset
    from audl_cv.cnn_lstm.inference import Inference
    
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

if __name__ == '__main__':
    print('hello world')
    run_test()