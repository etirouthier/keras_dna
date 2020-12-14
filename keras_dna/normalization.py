#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:40:31 2019

@author: routhier
"""

import pyBigWig
import re
import numpy as np


class Normalizer(object):
    """
    Normalize the data by standard procedure. It takes the wanted procedure and
    a bbi file as input. When called with a sequence extracted from the bbi
    file, the sequence is normalized.
        
    args:
        normalization:
            {zscore, max, perctrim, min_max, None}, name of the procedure to be
            applied.
        bbi_file:
            The file on which the data will be taken and then normalized.
        threshold:
            The pourcentage above which the distribution will be trimed.
    """ 
    def __init__(self,
                 normalization,
                 bbi_file,
                 threshold=99):
        self.normalization = normalization
        self.bbi_file = bbi_file
        self.bw = pyBigWig.open(bbi_file)
        self.threshold = threshold
        self.chrom_size = dict()

        for name, size in self.bw.chroms().items():
            if re.search('^(chr)?(\d+)?([XVI]+)?$', name):
                self.chrom_size[name] = size

        assert self.bw.isBigWig() or self.bw.isBigBed(),\
        """The file passed should be either a bigWig or a bigBed"""
        
        if self.normalization == 'zscore':
            self.mean = np.sum([size * self.bw.stats(name)[0]\
                                for name, size in self.chrom_size.items()])
            self.mean /= np.sum([size for name in self.chrom_size.values()])
            self.std = np.sum([size * self.bw.stats(name, type='std')[0] for name,
                               size in self.chrom_size.items()])
            self.std /= np.sum([size for name in self.chrom_size.values()])

        if self.normalization == 'max':
            self.max = np.max([self.bw.stats(name, type='max')[0]\
                               for name in self.chrom_size.keys()])

        if self.normalization == 'perctrim':
            sampled_values = get_sample(self.bw, self.chrom_size)
            self.limit = np.percentile(sampled_values, self.threshold)
            
        if self.normalization == 'min_max':
            self.max = np.max([self.bw.stats(name, type='max')[0]\
                               for name in self.chrom_size.keys()])
            self.min = np.min([self.bw.stats(name, type='min')[0]\
                               for name in self.chrom_size.keys()])
        self.bw.close()
    
    def __call__(self, seq):
        return self._normalize(seq)
        
    def _normalize(self, seq):
        if self.normalization == 'zscore':
            return (seq - self.mean) / self.std
        elif self.normalization == 'max':
            return (seq / self.max)
        elif self.normalization == 'perctrim':
            seq[seq > self.limit] = self.limit
            return seq
        elif self.normalization is None:
            return seq
        elif self.normalization == 'logtransform':
            return np.log(seq + 1)
        elif self.normalization == 'min_max':
            return (seq - self.min) / (self.max - self.min)
        
        
class BiNormalizer(Normalizer):
    
    def __init__(self,
                 normalization,
                 bbi_file,
                 threshold=99):
        assert len(normalization) == 2,\
        """BiNormalizer can only handle two successive normalization process"""
        self.normalization = normalization[0]
        self.threshold = threshold
        self.first_normalizer = Normalizer(normalization[1],
                                           bbi_file,
                                           self.threshold)
        self.bw = pyBigWig.open(bbi_file)
        self.sample = self.first_normalizer(get_sample(self.bw,
                                                       self.first_normalizer.chrom_size))
        self.bw.close()
        if self.normalization == 'zscore':
            self.mean = np.mean(self.sample)
            self.std = np.std(self.sample)

        if self.normalization == 'max':
            self.max = np.max(self.sample)

        if self.normalization == 'perctrim':
            self.limit = np.percentile(self.sample, self.threshold)
            
        if self.normalization == 'min_max':
            self.max = np.max(self.sample)
            self.min = np.min(self.sample)

    def __call__(self, seq):
        seq = self.first_normalizer(seq)
        return self._normalize(seq)


def get_sample(bw, chrom_size, sampling_len=1000):
    """Returns a sample of the values of the bbi_file as a numpy array"""
    sampled_values = []
    for name, size in chrom_size.items():
        sampling = np.random.randint(0, size - 1, sampling_len)
        sampled_values.append([bw.values(name, sample, sample + 1)[0]\
                               for sample in sampling])
    
    sampled_values =  np.array(sampled_values).reshape((len(sampled_values)*sampling_len,))
    return sampled_values[np.isfinite(sampled_values)]


class Weights(object):
    """
    info:
        doc: >
            Usefull to add weights to train a model. It calculate the probability
            of every labels that a generator will generate and can leads
            batches of weighting arrays corresponding to batches of labels.
     args:
        dataset:
            the dataset of the generator.
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
    """
    def __init__(self,
                 dataset,
                 weighting_mode=None,
                 bins='auto'):
        self.dataset = dataset
        self.command_dict = dataset.command_dict.get_details()

        if 'keras_dna.sequence.SparseDataset' in self.command_dict:
            try:
                df = dataset.dataset.df
            except AttributeError:
                df = dataset.seq_dl.dataset.df

            if weighting_mode == 'balanced':
                nb_neg = len(df[df.type ==  0])
                nb_pos = len(df[df.type != 0])
                
                self.value_positive = (nb_neg + nb_pos) / float(nb_pos * 2)
                self.value_negative = (nb_neg + nb_pos) / float(nb_neg * 2)

            if isinstance(weighting_mode, tuple):
                self.value_positive = weighting_mode[0]
                self.value_negative = weighting_mode[1]

        elif 'keras_dna.sequence.ContinuousDataset' in self.command_dict:
            try:
                chrom_size = dataset.dataset.chrom_size
                norm_dico = dataset.dataset.extractor.norm_dico
                self.tg_window = dataset.dataset.tg_window
            except AttributeError:
                chrom_size = dataset.seq_dl.dataset.chrom_size
                norm_dico = dataset.seq_dl.dataset.extractor.norm_dico
                self.tg_window = dataset.seq_dl.dataset.tg_window

            samples = list()
            annotation_files = self.command_dict['keras_dna.sequence.ContinuousDataset']['annotation_files']

            if not isinstance(annotation_files, list):
                annotation_files = [annotation_files]

            for file in annotation_files:
                bw = pyBigWig.open(file)
                samples.append(norm_dico[file](get_sample(bw, chrom_size)))
                bw.close()

            if weighting_mode == 'balanced':
                probas = list()
                self.list_bins = list()

                for sample in samples:
                    proba, bins = self._get_proba(sample, bins)
                    proba[proba == 0] = np.min(proba[proba > 0])
                    probas.append(proba)
                    self.list_bins.append(bins)

                self.weights = [1 / proba for proba in probas]
                self.weights = [weight / float(len(bins) - 1) for weight, bins\
                                in zip(self.weights, self.list_bins)]

            if isinstance(weighting_mode, tuple) or\
            isinstance(weighting_mode, list):
                try:
                    assert isinstance(weighting_mode[0], list)\
                    and isinstance(weighting_mode[1], list),\
                    """Weighting_mode should be 'balanced' or a tuple of list
                    or a tuple of numpy arrays"""
                except AssertionError:
                    assert isinstance(weighting_mode[0], np.ndarray)\
                    and isinstance(weighting_mode[1], np.ndarray),\
                    """Weighting_mode should be 'balanced' or a tuple of list
                    or a tuple of numpy arrays"""

                assert len(weighting_mode[0]) == len(annotation_files)\
                and len(weighting_mode[1]) == len(annotation_files),\
                """Weights and bins must be parsed for every annotation_files"""

                assert len(weighting_mode[0][0]) == len(weighting_mode[1][0]) + 1,\
                """len(bins) must be equal to len(weights) + 1 !"""

                self.weights = [weight for weight in weighting_mode[1]]
                self.list_bins = [bins for bins in weighting_mode[0]]

    def _get_proba(self, array, bins):
        counts, values = np.histogram(array,
                                      bins,
                                      range=(min(array) - 0.001, max(array)),
                                      density=True)
        return counts * (values[1] - values[0]), values

    def find_weights(self, seq):
        if 'keras_dna.sequence.SparseDataset' in self.command_dict:
            outputs = np.zeros((len(seq)))
            outputs += self.value_negative
            outputs[np.where(seq == 1)[0]] = self.value_positive
            return outputs
            
        elif 'keras_dna.sequence.ContinuousDataset' in self.command_dict:
            temporal = True
            if len(seq.shape) == 4:
                seq = seq.reshape((seq.shape[0],
                                   seq.shape[1],
                                   seq.shape[2] * seq.shape[3]))
            elif len(seq.shape) == 2:
                if seq.shape[1] == len(self.list_bins):
                    temporal = False
                    seq = np.expand_dims(seq, 1)
                elif seq.shape[1] == self.tg_window:
                    seq = np.expand_dims(seq, 2)
                else:
                    raise(ValueError("""The label are reshaped in a manner that
                    is not supported by 'balanced' keyword"""))

            outputs = np.zeros(seq.shape)

            for i in range(seq.shape[2]):
                digitized = np.digitize(seq[:, :, i],
                                        self.list_bins[i],
                                        right=True)
                digitized[digitized >= len(self.list_bins[i])] = len(self.list_bins[i]) - 1
                outputs[:, :, i] = np.asarray(self.weights[i])[digitized - 1]

            if temporal:
                return np.mean(outputs, axis=2)
            else:
                return np.mean(outputs, axis=2)[:, 0]
