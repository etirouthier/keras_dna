#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:21:30 2019

@author: routhier
"""

import numpy as np
import inspect

from sequence import SeqIntervalDl, StringSeqIntervalDl
from utils import ArgumentsDict

class Generator(object):
    """
    info:
        doc: >
            Generator for keras model able to yield inputs and targets in batch
            by reading into a fasta file and a annotation file. Inputs are one-
            hot-encoded or string.
     args:
         batch_size:
             number of example per batch pass to the model.
         one-hot-encoding:
             whether or not the inputs is one-hot-encoded (False: string)
             default: True
         args:
             arguments specific to the different dataloader that can be used.
         kwargs:
             dictionnary with specific arguments to the dataloader.
    """
    
    def __init__(self, batch_size,
                       one_hot_encoding=True,
                       *args,
                       **kwargs):
        self.one_hot_encoding = one_hot_encoding
        self.frame = inspect.currentframe()
        
        if self.one_hot_encoding:
            self.dataset = SeqIntervalDl(*args,
                                         **kwargs)
        else:
            self.dataset = StringSeqIntervalDl(*args,
                                               **kwargs)
        self.batch_size = batch_size

    def __call__(self):
        """Returns a generator to train a keras model (yielding inputs and
        outputs)."""
        def generator_function(dataset, batch_size):
            indexes = np.arange(0, len(dataset), 1)
            number_of_batches = len(dataset) // batch_size
            
            while True:
            # reshuffled the train set after an epoch
                np.random.shuffle(indexes)
                
                for num in range(number_of_batches):
                    batch_indexes = indexes[num*batch_size : (num + 1) * batch_size]
                    data = dataset[list(batch_indexes)]
                    yield data['inputs'], data['targets']
            
        return generator_function(self.dataset, self.batch_size)
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    @property
    def command_dict(self):
        return ArgumentsDict(self, called_args='dataset')

class MultiGenerator(object):
    """
    info:
        doc: >
            Class able to yield inputs and targets from several different
            interval readers. Usefull to train on several species or to train
            on both direct and reverse side.
     args:
         batch_size:
             number of example per batch pass to the model.
         dataset_list:
             list of SeqIntervalDl or StringSeqIntervalDl example, the output
             shape need to be the same for all instance.
        inst_per_dataset:
            list of integer, number of example to be taken from every dataset.
            default='all'
    """
    def __init__(self, batch_size,
                       dataset_list,
                       inst_per_dataset='all'):
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.inst_per_dataset = inst_per_dataset
        self.frame = inspect.currentframe()

    def __call__(self):
        """Returns a generator to train a keras model (yielding inputs and
        outputs)."""
        indexes = self._get_indexes()
        def generator_function(list_of_dataset, batch_size):
            number_of_batches = len(indexes) // batch_size
            
            while True:
            # reshuffled the train set after an epoch
                np.random.shuffle(indexes)
            
                for num in range(number_of_batches):
                    batch_indexes = indexes[num*batch_size : (num + 1) * batch_size]
                    inputs = None
                    targets = None

                    for dataset_index, dataset in enumerate(list_of_dataset): 
                        sub_batch_indexes = batch_indexes[batch_indexes[:, 0] == dataset_index]
                        data = list_of_dataset[dataset_index][sub_batch_indexes[:, 1].tolist()]
                        inputs = self._append_data(inputs, data['inputs'])
                        targets = self._append_data(targets, data['targets'])
                    yield inputs, targets
            
        return generator_function(self.dataset_list, self.batch_size)

    def _append_data(self, ldata, rdata):
        if isinstance(ldata, list):
            if len(rdata[0]) != 0:
                ldata[0] = np.append(ldata[0], rdata[0], axis=0)
                ldata[1] = np.append(ldata[1], rdata[1], axis=0)
        
        elif isinstance(ldata, np.ndarray):
            if len(rdata) != 0:
                ldata = np.append(ldata, rdata, axis=0)
        else:
            if isinstance(rdata, list):
                if len(rdata[0]) != 0:
                    ldata = rdata
            elif isinstance(rdata, np.ndarray):
                if len(rdata) != 0:
                    ldata = rdata
        return ldata

    def _get_indexes(self):
        if not self.inst_per_dataset == 'all':
            assert len(self.dataset_list) == len(self.inst_per_dataset),\
            """To pass the number of examples to be taken from every dataset,
            the list of dataset and list of number must be of the same length
            """

        indexes = np.zeros((1, 2))
        if self.inst_per_dataset == 'all':
            self.inst_per_dataset = [len(dataset) for dataset in self.dataset_list]
        
        for dataset_index, dataset in enumerate(self.dataset_list):
            length = len(dataset)
            nb_inst = self.inst_per_dataset[dataset_index]
            indexes_ = np.append(np.repeat([[dataset_index]], nb_inst,
                                           axis=0),
                                 np.random.choice(np.arange(length),
                                                  nb_inst,
                                                  replace=False).reshape(nb_inst,
                                                                         1),
                                 axis=1)
            
            indexes = np.append(indexes, indexes_, axis=0)
        indexes = indexes[1:].astype(int)
        return indexes

    def __len__(self):
        return len(self._get_indexes()) // self.batch_size
    
    @property
    def command_dict(self):
        return ArgumentsDict(self, kwargs=False)
