#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:21:30 2019

@author: routhier
"""

import numpy as np
import inspect
from copy import deepcopy

from .sequence import SeqIntervalDl, StringSeqIntervalDl
from .normalization import Weights
from .utils import ArgumentsDict, get_default_args

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
         output_shape:
             How to modify the shape of the output (because the initial output
             structure is (batch, length, nb_types, nb_annotation) or (batch,
             nb_types, nb_annotation))
         weighting_mode:
             {None, 'balanced', tuple(weights, bins), tuple(weight_pos, weight_neg)}
             the methodology to set the weights.
             For continuous dataset a tuple with the weights to apply and the
             bins can be parsed (len(weights) == len(bins) - 1, and the smallest
             bin must be smaller than the minimum of the data.)
             For sparse dataset a tuple with the weight to apply for positive
             (in [0]) and negative class (in [1]). Positive class refers to
             labels with at least one positive value.
             default: None
         bins:
             number of bins to apply to a continuous dataset before calculating
             the probability of classes. Can also be an array of bins or 'auto'
             for on optimized shearch of bins.
             default='auto'
         args:
             arguments specific to the different dataloader that can be used.
         kwargs:
             dictionnary with specific arguments to the dataloader.
    """
    def __init__(self, batch_size,
                       one_hot_encoding=True,
                       output_shape=None,
                       weighting_mode=None,
                       bins='auto',
                       *args,
                       **kwargs):
        self.one_hot_encoding = one_hot_encoding
        self.output_shape = output_shape
        self.weighting_mode = weighting_mode
        self.bins = bins
        self.frame = inspect.currentframe()

        old_shape = StringSeqIntervalDl.predict_label_shape(**kwargs)

        if self.output_shape:
            try:
                new_shape = self.output_shape[1:]
                np.zeros(old_shape).reshape(new_shape)
            except ValueError:
                raise ValueError("""output_shape is incompatible with the other arguments.
                Trying to reshape an array of shape {} in shape {}""".format(old_shape,
                                                                             new_shape))

        if self.one_hot_encoding:
            self.dataset = SeqIntervalDl(*args,
                                         **kwargs)
        else:
            self.dataset = StringSeqIntervalDl(*args,
                                               **kwargs)
        self.batch_size = batch_size
        
        if self.weighting_mode:
            self.weights = Weights(self.dataset,
                                   self.weighting_mode,
                                   self.bins)

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
                    inputs = data['inputs']
                    outputs = data['targets']
                    
                    if self.output_shape:
                        if isinstance(outputs, np.ndarray):
                            outputs = outputs.reshape((outputs.shape[0],) +\
                                                      tuple(self.output_shape)[1:])
                        else:
                            outputs[0] = outputs.reshape((outputs[0].shape[0],) +\
                                                          tuple(self.output_shape)[1:])
                    if self.weighting_mode:
                        weights = self.weights.find_weights(outputs)
                        
                        yield inputs, outputs, weights
                    else:
                        yield inputs, outputs
            
        return generator_function(self.dataset, self.batch_size)

    def __len__(self):
        return len(self.dataset) // self.batch_size + 1

    @property
    def command_dict(self):
        return ArgumentsDict(self, called_args='dataset')

    @classmethod
    def default_dict(cls):
        return get_default_args(cls.__init__)
    
    @classmethod
    def predict_sec_input_shape(cls, **input_dict):
        return StringSeqIntervalDl.predict_sec_input_shape(**input_dict)

    @property
    def secondary_input_shape(self):
        try:
            return tuple(self.dataset.secondary_input_shape)
        except AttributeError:
            return tuple(self.dataset.seq_dl.secondary_input)
    
    @classmethod
    def predict_label_shape(cls, **input_dict):
        command_dict = cls.default_dict()
        command_dict.update(input_dict)

        output_shape = StringSeqIntervalDl.predict_label_shape(**command_dict)

        if command_dict['output_shape']:
            try:
                new_shape = command_dict['output_shape'][1:]
                np.zeros(output_shape).reshape(new_shape)
                output_shape = new_shape
            except ValueError:
                raise ValueError("""output_shape is incompatible with the other arguments.
                Try to reshape an array of shape {} in shape {}""".format(output_shape,
                                                                          new_shape))
        return output_shape

    @property
    def label_shape(self):
        command_dict = self.command_dict.as_input()

        if command_dict['output_shape']:
            return tuple(command_dict['output_shape'][1:])
        else:
            return tuple(self.dataset.label_shape)

    @classmethod
    def predict_input_shape(cls, **input_dict):
        command_dict = cls.default_dict()
        command_dict.update(input_dict)
        
        if command_dict['one_hot_encoding']:
            return SeqIntervalDl.predict_input_shape(**command_dict)
        else:
            return None

    @property
    def input_shape(self):
        command_dict = self.command_dict.as_input()

        if command_dict['one_hot_encoding']:
            return tuple(self.dataset.input_shape)
        else:
            return None


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
             list of integer, number of example to be taken from each dataset.
             default='all'
         output_shape:
             How to modify the shape of the output (because the initial output
             structure is (batch, length, nb_types, nb_annotation) or (batch,
             nb_types, nb_annotation))
    """
    def __init__(self, batch_size,
                       dataset_list,
                       inst_per_dataset='all',
                       output_shape=None):
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.inst_per_dataset = inst_per_dataset
        self.output_shape = output_shape
        self.frame = inspect.currentframe()
        self._verify_dataset_list()

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
                        data = dataset[sub_batch_indexes[:, 1].tolist()]
                        inputs = self._append_data(inputs, data['inputs'])
                        targets = self._append_data(targets, data['targets'])
                    if self.output_shape:
                        targets = targets.reshape((targets.shape[0],) +\
                                                   tuple(self.output_shape)[1:])
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
        argsdict_list = [dataset.command_dict for dataset in self.dataset_list]
        argsdict_list.append(ArgumentsDict(self, kwargs=False))
        return argsdict_list

    def _verify_dataset_list(self):
        double_check = zip(self.dataset_list[:-1],
                           self.dataset_list[1:])

        for d1, d2 in double_check:
            assert type(d1) == type(d2),\
            """All dataset should be of the same class"""

            try:
                assert d1.input_shape == d2.input_shape,\
                """Found incompatible input shape: {} and {}""".format(d1.input_shape,
                                                                       d2.input_shape)
            except NameError:
                pass

            assert d1.secondary_input_shape == d2.secondary_input_shape,\
            """Found incompatible secondary input shape: {} and {}""".\
            format(d1.secondary_input_shape, d2.secondary_input_shape)

            assert d1.label_shape == d2.label_shape,\
            """Found incompatible label shape {} and {}""".format(d1.label_shape,
                                                                  d2.label_shape)

        if self.output_shape:
            try:
                np.zeros(self.output_shape[1:]).reshape(self.dataset_list[0].label_shape)
            except ValueError:
                raise ValueError("""output_shape is incompatible with the other arguments.
                Trying to reshape an array of shape {} in shape {}""".\
                                 format(self.output_shape[1:],
                                        self.dataset_list[0].label_shape))

    @property
    def input_shape(self):
        try:
            return self.dataset_list[0].input_shape
        except NameError:
            return None

    @property
    def secondary_input_shape(self):
        return self.dataset_list[0].secondary_input_shape

    @property
    def label_shape(self):
        if self.output_shape:
            return self.output_shape[1:]
        else:
            return self.dataset_list[0].label_shape 


class PredictionGenerator(object):
    """
    info:
        doc: >
            Takes the command dict of a Generator instance (or a SeqIntervalDl
            or a StringSeqIntervalDl instance) and returns a generator needed
            to predict all along chromosomes.
    
    args:
        batch_size:
            integer
        chrom_size:
            A .sizes file with the size of chromosomes in two columns.
        command_dict:
            An ArgumentsDict instance that described how the model was trained.
        incl_chromosomes:
            list of chromosome to predict on.
        start_stop:
            list of tuple indicating where to start and stop generating data.
            One per included chromosome. default=None
        fasta_file:
            A fasta_file to make prediction on (if None the same as in
            command_dict)
            default=None
        rc:
            Weither or not to predict with reverse complemented DNA sequences.
            default=False
        overlapping:
            int or False, if int the intervals are overalpped with a stride of
            int.
            default=False
    """
    def __init__(self,
                 batch_size,
                 command_dict,
                 chrom_size,
                 incl_chromosomes,
                 start_stop=None,
                 fasta_file=None,
                 rc=False,
                 overlapping=False):
        self.batch_size = batch_size
        self.command_dict = command_dict
        self.chrom_size = chrom_size
        self.start_stop = start_stop
        self.rc = rc
        self.overlapping = overlapping

        if isinstance(incl_chromosomes, list):
            self.incl_chromosomes = incl_chromosomes
        else:
            self.incl_chromosomes = [incl_chromosomes]
        self.detailed_dict = command_dict.get_details()

        if 'keras_dna.sequence.SparseDataset' in self.detailed_dict:
            dataset_dict = self.detailed_dict['keras_dna.sequence.SparseDataset']
            self.window = dataset_dict['length']
            self.tg_window = 1

            if dataset_dict['seq2seq']:
                self.tg_window = self.window

        elif 'keras_dna.sequence.ContinuousDataset' in self.detailed_dict:
            dataset_dict = self.detailed_dict['keras_dna.sequence.ContinuousDataset']
            self.window = dataset_dict['window']

            if dataset_dict['downsampling']:
                self.tg_window = dataset_dict['window']
                self.sampling_len = dataset_dict['window'] // dataset_dict['tg_window']
            else:
                self.tg_window = dataset_dict['tg_window']

        string_dict = deepcopy(self.detailed_dict['keras_dna.sequence.StringSeqIntervalDl'])
        string_dict['annotation_files'] = self.chrom_size
        string_dict['use_strand'] = False
        string_dict['rc'] = self.rc
        if fasta_file:
            string_dict['fasta_file'] = fasta_file

        continuous_dict = {'ignore_targets' : True,
                           'overlapping' : self.overlapping,
                           'window' : self.window,
                           'incl_chromosomes' : self.incl_chromosomes,
                           'start_stop' : self.start_stop,
                           'tg_window' : self.tg_window}

        if 'keras_dna.sequence.SeqIntervalDl' in self.detailed_dict:
            self.input_dict = deepcopy(self.detailed_dict['keras_dna.sequence.SeqIntervalDl'])
            self.input_dict.update(string_dict)
            self.input_dict.update(continuous_dict)
            self.dataset = SeqIntervalDl(**self.input_dict)
        else:
            self.input_dict = string_dict
            self.input_dict.update(continuous_dict)
            self.dataset = StringSeqIntervalDl(**self.input_dict)

    def __len__(self):
        return len(self.dataset) // self.batch_size + 1

    def __call__(self):
        """Returns a generator to train a keras model (yielding inputs and
        outputs)."""
        def generator_function(dataset, batch_size):
            indexes = np.arange(len(dataset))
            number_of_batches = len(dataset) // batch_size + 1

            while True:
                for num in range(number_of_batches):
                    if num == number_of_batches - 1:
                        batch_indexes = indexes[num * batch_size :]
                    else:
                        batch_indexes = indexes[num*batch_size : (num + 1) * batch_size]
                    data = dataset[list(batch_indexes)]
                    yield data['inputs']

        return generator_function(self.dataset, self.batch_size)

    @property
    def index_df(self):
        """
        Returns a DataFrame with the first and last index for every chromosome
        and the position of the begining (included) of the prediction and the
        end (excluded) of the prediction on the chromosome.
        """
        if 'keras_dna.sequence.SeqIntervalDl' in self.detailed_dict:
            df = deepcopy(self.dataset.seq_dl.dataset.df)
        else:
            df = deepcopy(self.dataset.dataset.df)

        if 'keras_dna.sequence.SparseDataset' in self.detailed_dict:
            dataset_dict = self.detailed_dict['keras_dna.sequence.SparseDataset']

            if dataset_dict['seq2seq']:
                df['start'] = df.start.values - self.window // 2
                df['stop'] = df.start.values + (df.last_index.values - df.first_index.values)\
                            * self.window + self.window
            else:
                df['start'] = df.start.values
                df['stop'] = df.start.values + (df.last_index.values - df.first_index.values)\
                            + 1
        elif 'keras_dna.sequence.ContinuousDataset' in self.detailed_dict:
            dataset_dict = self.detailed_dict['keras_dna.sequence.ContinuousDataset']

            if dataset_dict['downsampling']:
                df['start'] = df.start.values - self.window // 2
                df['stop'] = df.start.values + (df.last_index.values - df.first_index.values)\
                            * self.window + self.window
            else:
                tg_window = dataset_dict['tg_window']
                df['start'] = df.start.values - tg_window // 2
                df['stop'] = df.start.values + (df.last_index.values - df.first_index.values)\
                            * tg_window + tg_window
        return df
