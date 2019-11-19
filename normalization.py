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
            if re.search('^(chr)?\d+$', name):
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


def get_sample(bw, chrom_size):
    """Returns a sample of the values of the bbi_file as a numpy array"""
    sampling_len = 1000
    sampled_values = []
    for name, size in chrom_size.items():
        sampling = np.random.randint(0, size - 1, sampling_len)
        sampled_values.append([bw.values(name, sample, sample + 1)[0]\
                               for sample in sampling])
    
    sampled_values =  np.array(sampled_values).reshape((len(sampled_values)*sampling_len,))
    return sampled_values[np.isfinite(sampled_values)]
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            