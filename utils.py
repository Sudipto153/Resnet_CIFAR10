import torch
import pandas as pd
import numpy as np
import math
import time
import json
from collections import OrderedDict, namedtuple
from itertools import product
from IPython.display import display, clear_output


class RunBuilder():
    @staticmethod
    def get_runs(params):
        
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
            
        return runs


class RunManager():
    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None


    def begin_run(self, run, network, loader):
        
        self.run_start_time = time.time()
        
        self.run_params = run
        self.run_count += 1
        
        self.network = network
        self.loader = loader

        
    def end_run(self):
        self.epoch_count = 0
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        
    def end_epoch(self):
        
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
        loss = self.epoch_loss/len(self.loader.dataset)
        accuracy = self.epoch_num_correct/len(self.loader.dataset)
            
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')
        
        clear_output(wait = True)
        display(df)
        
    def track_loss(self, loss):
        self.epoch_loss += loss.item()*self.loader.batch_size
        #self.epoch_loss += loss.item()*500

        
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)
        
    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim = 1).eq(labels).sum().item()
    
    def save(self, filename):
        
        pd.DataFrame.from_dict(
            self.run_data,
            orient = 'columns'
        ).to_csv(f'{filename}.csv')
        
        with open(f'{filename}.json','w', encoding = 'utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii = False, indent = 4)