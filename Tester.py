import numpy as np 
import torch
from torch.autograd import Variable

def predict(net, test_dataloader):
    print("length: ", len(test_dataloader))
    pred_store = []
    true_store = []
    with torch.no_grad():
      for batch_idx, (data,target) in enumerate(test_dataloader):
        data, target = Variable(data), Variable(target).float()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = data.to(device)
        pred = net(data) #batch x c x 40 x 128 x 128 prediction 
        data = data.cpu()
        pred_store.append(np.argmax(pred.detach().cpu().numpy(), axis = 1) ) 
        true_store.append(np.argmax(target.detach().cpu().numpy(), axis=1))
    

    pred_store = np.array(pred_store)
    true_store = np.array(true_store) 
    #I think here I'm flattening the batches so that it has shape
    #N x 40 x 128 x 128 
    #A.reshape(A.shape[:-1] + (32,3))
    if pred_store.ndim == 5:
        pred_store = pred_store.reshape(pred_store.shape[0]*pred_store.shape[1], pred_store.shape[2], pred_store.shape[3], pred_store.shape[4])
        true_store = true_store.reshape(true_store.shape[0]*true_store.shape[1], true_store.shape[2], true_store.shape[3], true_store.shape[4])
    else: #if pred_store.ndim == 4
        pred_store = pred_store.reshape(pred_store.shape[0]*pred_store.shape[1], pred_store.shape[2], pred_store.shape[3])
        true_store = true_store.reshape(true_store.shape[0]*true_store.shape[1], true_store.shape[2], true_store.shape[3])

    return pred_store, true_store