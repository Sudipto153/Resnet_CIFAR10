import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from collections import OrderedDict

from model import Network
from loader_splitter import get_test_loader
from utils import*



def main():

    ## test set loading
    test_loader, classes = get_test_loader(
                            root_dir = './datasets',
                            batch_size = 128,
                            augmented = True)


    ## load network
    PATH_test = 'networks/network.pt'
    nett = Network().cuda()
    nett.load_state_dict(torch.load(PATH_test))
    nett.eval()

    ##nett = network
    device_t = torch.device('cuda')
    total_loss = 0
    total_correct = 0

    for batch_t in test_loader:
        
        images = batch_t[0].to(device_t)
        labels = batch_t[1].to(device_t)
        preds_t = nett(images)  ## pass batch
        loss_t = F.cross_entropy(preds_t, labels) 
        
        total_loss += loss_t.item()*test_loader.batch_size
        total_correct += preds_t.argmax(dim = 1).eq(labels).sum().item()
        
    print(total_loss)
    print('accuracy:', total_correct/(len(test_loader)*test_loader.batch_size))


if __name__ == "__main__":
    main()