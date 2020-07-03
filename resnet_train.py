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
from loader_splitter import get_data_loaders
from utils import*



def main():

    ## loading training and dev data
    train_loader, dev_loader = get_data_loaders(
                                root_dir = './datasets',
                                batch_size = 128,
                                augmented = True)


    ## training the network
    params = OrderedDict(
        lr = [0.0009],
        device = ['cuda']
    )

    num_epochs = 200
    decay_rate = 0.96
    m = RunManager()
    for run in RunBuilder.get_runs(params):

        device = torch.device(run.device)
        network = Network().to(device)
        loader = train_loader
        optimizer = optim.Adam(network.parameters(), lr = run.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = decay_rate)

        m.begin_run(run, network, loader)
        for epoch in range(num_epochs):
            m.begin_epoch()
            
            for batch in loader:        ## get batch
                
                images = batch[0].to(device)
                labels = batch[1].to(device)
                preds = network(images)  ## pass batch
                loss = F.cross_entropy(preds, labels)   ## calculate loss

                optimizer.zero_grad()
                loss.backward()  ## calculate gradients
                optimizer.step()  #update weights
                
                m.track_loss(loss)
                m.track_num_correct(preds, labels)
            
            m.end_epoch()
            scheduler.step()
        
        m.end_run()

    m.save('results')


    ## saving the network
    PATH = 'networks/network.pt'
    torch.save(network.state_dict(), PATH)


    ## dev_set_testing
    PATH_dev = 'networks/network.pt'
    net = Network().cuda()
    net.load_state_dict(torch.load(PATH_dev))
    net.eval()

    device_test = torch.device('cuda')
    total_loss = 0
    total_correct = 0

    for batch_test in dev_loader:
        
        images = batch_test[0].to(device_test)
        labels = batch_test[1].to(device_test)
        preds_test = net(images)  ## pass batch
        loss_test = F.cross_entropy(preds_test, labels) 
        
        total_loss += loss_test.item()*dev_loader.batch_size
        total_correct += preds_test.argmax(dim = 1).eq(labels).sum().item()
        
    print(total_loss)
    print('accuracy:', total_correct/(len(dev_loader)*dev_loader.batch_size))


if __name__ == "__main__":
    main()