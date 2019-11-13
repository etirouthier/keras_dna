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

        
    def __init__(self,
                 normalization,
                 bbi_file):
        self.normalization = normalization
        self.bw = pyBigWig.open(bbi_file)
        self.chrom_size = dict()
        for name, size in self.bw.chroms().items():
            if re.search('^(chr)?\d+$', name):
                self.chrom_size[name] = size
        
        assert self.bw.isBigWig() or self.bw.isBigBed(),\
        """The file passed should be either a bigWig or a bigBed"""
        
        if self.normalization == 'zscore':
            self.mean = np.sum([size * self.bw.stats(name)[0] for name, size in self.chrom_size.items()])
            self.mean /= np.sum([size for name in self.chrom_size.values()])
            self.std = np.sum([size * self.bw.stats(name, type='std')[0] for name, size in self.chrom_size.items()])
            self.std /= np.sum([size for name in self.chrom_size.values()])
            self.bw.close()
        
    def normalize(self, seq):
        
        if self.normalization == 'zscore':
            return (seq - self.mean) / self.std 
        
        elif self.normalization is None:
            return seq