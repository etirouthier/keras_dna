#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:37:22 2019

@author: routhier
"""

import numpy as np
import pyBigWig
import warnings


from kipoiseq.transforms.functional import resize_interval


from .utils import rolling_window
from .normalization import Normalizer, BiNormalizer


class bbi_extractor(object):
    """
    Reads the data into a bigWig file. Returns the coverage on an interval
    with the desired normalization and sampling strategy.

    args:
        bbi_files:
            List of bbi_files from which the data will be taken.
        window:
            The desired length of the output window.
        nb_annotation_type:
            The number of different annotation in the bbi_files. The same
            number of files must be passed for every annotation, the list must
            be organised as [file1_ann1, file1_ann2, file2_ann1, file2_ann2],
            with file1, file2 designing two differents kind of files (different
            lab, different cellular type ...). 
            If None the output shape will be (window, nb_of_files).
        sampling_mode:
            {None, 'mean', 'downsampling'}, the sampling strategy to apply
            at the coverage.
        normalization_mode:
            argument from the Normalizer class
        *args, **kwargs:
            other arguments from Normalizer class
    """
    def __init__(self,
                 bbi_files,
                 window,
                 nb_annotation_type=None,
                 sampling_mode=None,
                 normalization_mode=None,
                 *args,
                 **kwargs):
        if not isinstance(bbi_files, list):
            self.bbi_files = [bbi_files]
        else:
            self.bbi_files = bbi_files

        self.window = window
        self.nb_annotation_type = nb_annotation_type
        self.sampling_mode = sampling_mode
        self.normalization_mode = normalization_mode

        #TODO create a bam, bedGraph, wig to bigWig converter
        self.norm_dico = dict()
        if isinstance(self.normalization_mode, list):
            for bbi_file in self.bbi_files:
                self.norm_dico[bbi_file] = BiNormalizer(normalization_mode,
                                                        bbi_file,
                                                        *args,
                                                        **kwargs)
        else:
            for bbi_file in self.bbi_files:
                self.norm_dico[bbi_file] = Normalizer(normalization_mode,
                                                      bbi_file,
                                                      *args,
                                                      **kwargs)

        if self.nb_annotation_type:
            assert len(self.bbi_files) % self.nb_annotation_type == 0,\
            """Every annotation must be described by the same number of file"""
            
            warnings.warn("""the list must be organised as [file1_ann1, file1_ann2,
            file2_ann1, file2_ann2], with file1, file2 designing two differents kind
            of files (different lab, different cellular type ...)""")

    def extract(self, interval):
        """
        Extract the coverage corresponding to the interval from the bbi_files.
        
        returns:
            np.array of shape (window,
                               number of files per annotation,
                               number of annotation)
        """
        if not self.sampling_mode:
            assert self.window <= abs(interval.stop - interval.start),\
            """The target window must be smaller than the input length"""

            interval = resize_interval(interval,
                                       self.window,
                                       anchor='center')
        seq = list()
        for bbi_file in self.bbi_files:
            bw = pyBigWig.open(bbi_file)
            array = bw.values(interval.chrom,
                              interval.start,
                              interval.stop, numpy=True)
            array[np.isnan(array)] = 0
            seq.append(self.norm_dico[bbi_file](array))
        seq = np.array(seq).T

        if self.sampling_mode:
            assert abs(interval.stop - interval.start) % self.window == 0,\
            """Window must divide the input length to use downsampling"""
            sampling_length = abs(interval.stop - interval.start) // self.window

            if self.sampling_mode == 'downsampling':
                    seq = seq[::sampling_length]
            elif self.sampling_mode == 'mean':
                    seq = self._calculate_rolling_mean(seq)
            else:
                raise NameError('sampling_mode must be None, "mean" or "downsampling"')
        
        if self.nb_annotation_type:
            nb_files_per_ann = len(self.bbi_files) // self.nb_annotation_type
            return seq.reshape((self.window,
                                nb_files_per_ann,
                                self.nb_annotation_type))
        
        else:
            return seq

    def _calculate_rolling_mean(self, x):
        sampling_length = len(x) // self.window
        num_classes = x.shape[1]
        x = rolling_window(x,
                           window=(sampling_length, num_classes),
                           asteps=(sampling_length, num_classes))
        x = x.reshape((self.window, sampling_length, num_classes))
        x = np.mean(x, axis=1, dtype='float32')
        return x