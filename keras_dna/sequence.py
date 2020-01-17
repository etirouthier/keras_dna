#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:17:41 2019

@author: routhier
"""

import pandas as pd
import numpy as np
import pybedtools
import random
import warnings
import pyBigWig
import inspect


from kipoi.metadata import GenomicRanges
from kipoi.data import Dataset
from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.transforms import ReorderedOneHot
from kipoiseq.transforms.functional import fixed_len
from kipoiseq.utils import DNA


from . import utils
from .extractors import bbi_extractor


class SparseDataset(object):
    """
    info:
        docs: >
            Reads the positions corresponding to some annotations in a file
            dedicated to store sparse annotation (gff, gtf, bed) and return a
            pybedtool interval corresponding to every annitation as long as a
            label for every interval.
    args:
        annotation_files:
            list of file with annotations (one file per cellular type for
            example, several annotation per file is possible)
        annotation_list:
            list of annotation to be taken into account (name of those
            annotation in the files)
        predict:
            {'all', 'start', 'stop'} weither to predict the annotation, its
            start or its end, default='all'
        seq_len:
            {'MAXLEN', int, 'real'} the length of the intervals. If MAXLEN then
            the length will be the maximal annotation length in the files, real
            mean that the row intervals are returned.
            default='MAXLEN'
        data_augmentation:
            boolean, if true return all the window of the given length where an
            annotation fit entirely in (false = one window per annotation),
            default=False
        seq2seq:
            boolean, if true the label will be of the length of the input
            sequence with 1 where the annotations are in this sequence,
            default=False
        define_positive:
            {'match_all', 'match_any'} if not seq2seq a sequence will be
            considered positive to an annotation if it matches all or any of an
            annotation instance, default='match_all'.
        num_chr:
            if specified, 'chr' in the chromosome name will be dropped,
            default=False
        incl_chromosomes:
            exclusive list of chromosome names to include in the final dataset.
            if not None, only these will be present in the dataset,
            default=None
        excl_chromosomes:
            list of chromosome names to omit from the dataset. default=None
        ignore_targets: 
            if True, target variables are ignored, default=False
        negative_ratio:
            'all' or int, ratio of negative example compared to positive,
            'all' means that all the negative example are returned.
            default=1
        negative_type:
            {'real', 'random', None} if real the negative example will be taken
            from sequences far enough from any annotation example. If None this
            function will return only positive example, 'random' will return 
            interval of length 0.
            default='real'
    """
    def __init__(self, annotation_files,
                       annotation_list,
                       predict='all',
                       seq_len='MAXLEN',
                       data_augmentation=False,
                       seq2seq=False,
                       defined_positive='match_all',
                       num_chr=False,
                       incl_chromosomes=None,
                       excl_chromosomes=None,
                       ignore_targets=False,
                       negative_ratio=1,
                       negative_type='real'):
        self.annotation_files = annotation_files
        self.annotation_list = annotation_list
        self.predict = predict
        self.seq_len = seq_len
        self.data_augmentation = data_augmentation
        self.seq2seq = seq2seq
        self.defined_positive = defined_positive
        self.num_chr = num_chr
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.ignore_targets = ignore_targets
        self.negative_ratio = negative_ratio
        self.negative_type = negative_type
        self.frame = inspect.currentframe()

        assert not (self.seq_len == 'real' and self.data_augmentation), \
        '''Returning the real position of the annotation is not compatible with
        data_augmentation'''

        if not isinstance(self.annotation_files, list):
            self.annotation_files = [self.annotation_files]

        df_ann_list = list()

        for annotation_file in self.annotation_files:
            if annotation_file.endswith('.bed'):
                df_ann_list.append(utils.bed_to_df(annotation_file,
                                                   self.annotation_list))
            if annotation_file.endswith(('.gff', 'gff3', 'gtf')):
                df_ann_list.append(utils.gff_to_df(annotation_file,
                                                   self.annotation_list))

        self.ann_df = self._multi_cellular_type(df_ann_list)
        self._binarize_label()
        
        if self.num_chr and self.ann_df.iloc[0][0].startswith("chr"):
            self.ann_df['chrom'] = self.ann_df['chrom'].str.replace("^chr", "")
        if not self.num_chr and not self.ann_df.iloc[0][0].startswith("chr"):
            self.ann_df['chrom'] = "chr" + self.ann_df['chrom']
        
        # omit data outside chromosomes
        if incl_chromosomes is not None:
            self.ann_df = self.ann_df[self.ann_df.chrom.isin(incl_chromosomes)]
        if excl_chromosomes is not None:
            self.ann_df = self.ann_df[~self.ann_df.chrom.isin(excl_chromosomes)]
        
        if not self.predict == 'all':
            self._restrict()
        
        if self.seq_len == 'MAXLEN' or self.seq_len == 'real':
            self.length = self._find_maxlen()
        elif isinstance(self.seq_len, int):
            self.length = self.seq_len
        else:
            raise NameError('seq_len should be "MAXLEN", "real" or an integer')

        self.df = self._get_dataframe()

        if not self.ignore_targets:
            self.labels = self._get_labels()

        if self.negative_type == 'random':
            assert isinstance(self.negative_ratio, int), \
            'To use random negative sequence negative_ratio must be an integer'
            self._random_negative_class()

        elif self.negative_type == 'real':
            neg_df, neg_label = self._negative_class()
            self.df = self.df.append(neg_df)

            if not self.ignore_targets:
                self.labels = np.append(self.labels, neg_label, axis=0)

    def __getitem__(self, idx):
        """Returns (pybedtools.Interval, labels)"""
        if not isinstance(idx, list):
            idx = [idx]
        row = self.df.iloc[idx]
        
        if self.ignore_targets:
            labels = {}
        else:
            labels = self.labels[idx]
            index = []

        intervals = list()
        
        if 'strand' in self.df.columns:
            for i in range(len(idx)):
                row_ = row.iloc[i]
                try:
                    intervals.append(pybedtools.create_interval_from_list([row_.chrom,
                                                                           int(row_.start),
                                                                           int(row_.stop),
                                                                           '.', '.',
                                                                           row_.strand]))
                    if not self.ignore_targets:
                        index.append(i)

                except OverflowError:
                    warnings.warn("""Some of the input sequence were out of range
                                  and have been removed""")

        else:
            for i in range(len(idx)):
                row_ = row.iloc[i]
                try:
                    intervals.append(pybedtools.create_interval_from_list([row_.chrom,
                                                                           int(row_.start),
                                                                           int(row_.stop)]))
                    if not self.ignore_targets:
                        index.append(i)

                except OverflowError:
                    warnings.warn("""Some of the input sequence were out of range
                                  and have been removed""")
        return intervals, labels[index]

    def __len__(self):
        return len(self.df)

    def _multi_cellular_type(self, list_df):
        multi_df = pd.DataFrame()
        for i, data in enumerate(list_df):
            data['type'] = i + 1
            multi_df = multi_df.append(data)
        return multi_df

    def _binarize_label(self):
        self.ann_df = self.ann_df[self.ann_df.label.isin(self.annotation_list)]
        labels = self.ann_df.label.values

        for i, label in enumerate(self.annotation_list):
            labels[labels == label] = i + 1
        self.ann_df.label = labels

    def _restrict(self):
        assert 'strand' in self.ann_df.columns,\
        'The data need to specify the strand to use restrict'
        df = self.ann_df
        df_plus = pd.DataFrame()
        df_minus = pd.DataFrame()
        if self.predict == 'start':
            df_plus['chrom'] = df[df.strand == '+'].chrom.values
            df_plus['start'] = df[df.strand == '+'].start.values
            df_plus['stop'] = df[df.strand == '+'].start.values + 1
            df_plus['label'] = df[df.strand == '+'].label.values
            df_plus['strand'] = '+'
            df_plus['type'] = df[df.strand == '+'].type.values
            
            df_minus['chrom'] = df[df.strand == '-'].chrom.values
            df_minus['start'] = df[df.strand == '-'].stop.values - 1
            df_minus['stop'] = df[df.strand == '-'].stop.values
            df_minus['label'] = df[df.strand == '-'].label.values
            df_minus['strand'] = '-'
            df_minus['type'] = df[df.strand == '-'].type.values

        elif self.predict == 'stop':
            df_plus['chrom'] = df[df.strand == '+'].chrom.values
            df_plus['start'] = df[df.strand == '+'].stop.values - 1
            df_plus['stop'] = df[df.strand == '+'].stop.values
            df_plus['label'] = df[df.strand == '+'].label.values
            df_plus['strand'] = '+'
            df_plus['type'] = df[df.strand == '+'].type.values
            
            df_minus['chrom'] = df[df.strand == '-'].chrom.values
            df_minus['start'] = df[df.strand == '-'].start.values
            df_minus['stop'] = df[df.strand == '-'].start.values + 1
            df_minus['label'] = df[df.strand == '-'].label.values
            df_minus['strand'] = '-'
            df_minus['type'] = df[df.strand == '-'].type.values
            
        self.ann_df = df_plus.append(df_minus)
    
    def _find_maxlen(self):
        return np.max(self.ann_df.stop.values - self.ann_df.start.values)

    def _calculate_interval(self,
                            df,
                            return_all=False,
                            return_strand=False):
        assert (df.stop.values - df.start.values <= self.length).any(),\
        'The size of the window need to be greater than every annotation instance'
        if return_strand:
            assert 'strand' in df.columns, \
            'To return the strand the dataframe should have a strand column'
    
        start = df.start.values
        stop = df.stop.values

        if self.data_augmentation and return_all:
            starts = np.concatenate([np.arange(stop[i] - self.length,
                                               start[i] + 1)\
                                     for i in range(len(start))],
                                     axis=0)
            stops = np.concatenate([np.arange(stop[i],
                                              start[i] + 1 + self.length)\
                                    for i in range(len(start))],
                                    axis=0)

            if return_strand:
                strand = df.strand.values
                strands = np.concatenate([np.repeat(strand[i],
                                                    start[i] - stop[i] + self.length + 1)\
                                          for i in range(len(strand))])
                return starts, stops, strands
            else:
                return starts, stops

        elif self.data_augmentation and not return_all:
            return stop - self.length, start + self.length
    
        else:
            wx = (self.length - (stop - start))
            half_wx = wx // 2

            if return_strand:
                return start - half_wx - wx % 2, stop + half_wx, df.strand.values
            else:
                return start - half_wx - wx % 2, stop + half_wx

    def _random_negative_class(self):
        chrom = self.df.chrom.unique()[0]
        number_neg = self.negative_ratio * len(self.df)

        neg_df = pd.DataFrame()
        neg_df['start'] = np.zeros((number_neg,))
        neg_df['stop'] = np.zeros((number_neg,))
        neg_df['chrom'] = chrom
        
        if 'strand' in self.ann_df.columns:
            neg_df['strand'] = np.random.choice(['+', '-'], number_neg)
            
        self.df = self.df.append(neg_df)

        if not self.ignore_targets:
            neg_shape = list(self.labels.shape)
            neg_shape[0] = number_neg
            neg_label = np.zeros(tuple(neg_shape))

            self.labels = np.append(self.labels,
                                    neg_label,
                                    axis=0)

    def _negative_class(self):
        neg_df = pd.DataFrame()
        number_of_pos = list()
    
        for chrom in self.ann_df.chrom.unique():
            neg_starts = np.arange(1,
                                   np.max(self.ann_df[self.ann_df.chrom == chrom].stop.values) + 1)
            df_ = self.ann_df[self.ann_df.chrom == chrom]
            pos_starts, pos_stops = self._calculate_interval(df_)
            number_of_pos.append(np.sum(pos_stops - pos_starts))
    
            pos_starts = pos_starts - self.length
            deletion = np.concatenate([np.arange(pos_starts[i], pos_stops[i])\
                                       for i in range(len(pos_starts))],
                                      axis=0)
            deletion = deletion[deletion >= 0]
            neg_starts = np.delete(neg_starts, deletion)
    
            neg_df_ = pd.DataFrame()
            neg_df_['start'] = neg_starts
            neg_df_['stop'] = neg_starts + self.length
            neg_df_['chrom'] = chrom

            neg_df = neg_df.append(neg_df_)

        neg_df['label'] = 0
        neg_df['type'] = 0

        if 'strand' in self.ann_df.columns:
            neg_df['strand'] = np.random.choice(['+', '-'], len(neg_df))
        
        nb_types = len(self.ann_df.type.unique())
        nb_labels = len(self.ann_df.label.unique())
        
        if self.negative_ratio == 'all':
            if self.seq2seq:
                labels = np.zeros((len(neg_df),
                                   self.length,
                                   nb_types,
                                   nb_labels))
            else:
                labels = np.zeros((len(neg_df),
                                   nb_types,
                                   nb_labels))
            
            return neg_df, labels
        
        elif isinstance(self.negative_ratio, int):
            if self.data_augmentation:
                indexes = np.random.choice(np.arange(len(neg_df)),
                                 sum(number_of_pos) * self.negative_ratio,
                                 replace=False)
            else:
                indexes = np.random.choice(np.arange(len(neg_df)),
                                 len(self.ann_df) * self.negative_ratio,
                                 replace=False)

            if self.seq2seq:
                labels = np.zeros((len(indexes),
                                   self.length,
                                   nb_types,
                                   nb_labels))
            else:
                labels = np.zeros((len(indexes),
                                   nb_types,
                                   nb_labels))
            
            return neg_df.iloc[indexes], labels
        else:
            raise NameError('negative_ratio should be "all" or an integer')

    def _get_dataframe(self):
        new_df = pd.DataFrame()
        
        for chrom in self.ann_df.chrom.unique():
            df_ = self.ann_df[self.ann_df.chrom == chrom]
            
            if not self.seq_len == 'real':
                if 'strand' in df_.columns:
                    pos_starts, pos_stops, pos_strands = \
                    self._calculate_interval(df_,
                                             return_all=True,
                                             return_strand=True)
                    
                else:
                    pos_starts, pos_stops = \
                    self._calculate_interval(df_,
                                             return_all=True)
                
                new_df_ = pd.DataFrame({'start' : pos_starts,
                                        'stop' : pos_stops})
                new_df_['chrom'] = chrom
    
                if 'strand' in df_.columns:
                    new_df_['strand'] = pos_strands

                new_df = new_df.append(new_df_)
            else:
                new_df = new_df.append(df_)
        return new_df

    def _get_labels(self):
        nb_types = len(self.ann_df.type.unique())
        nb_labels = len(self.ann_df.label.unique())

        if self.seq2seq:
            labels = np.zeros((1, self.length, nb_types, nb_labels))
        else:
            labels = np.zeros((1, nb_types, nb_labels))

        for chrom in self.ann_df.chrom.unique():
            df_ = self.ann_df[self.ann_df.chrom == chrom]
            pos_starts, pos_stops = self._calculate_interval(df_,
                                                             return_all=True)
            if self.seq2seq:
                labels_ = np.zeros((len(pos_starts),
                                    self.length,
                                    nb_types,
                                    nb_labels))
            else:
                labels_ = np.zeros((len(pos_starts),
                                    nb_types,
                                    nb_labels))
            for label in np.unique(df_.label.values):
                for cell_type in np.unique(df_.type.values):
                    local_df = df_[df_.label == label]
                    local_df = local_df[local_df.type == cell_type]

                    SIZE = 10000
                    for first_idx in np.append(np.arange(0, len(pos_starts), SIZE),
                                               np.array(len(pos_starts))):

                        pos_starts_ = pos_starts[first_idx : first_idx + SIZE]
                        pos_stops_ = pos_stops[first_idx : first_idx + SIZE]

                        mat_ann_start = np.repeat(local_df.start.values,
                                                  len(pos_starts_)).\
                                                  reshape(len(local_df),
                                                          len(pos_starts_)).T
                        mat_int_start = np.repeat(pos_starts_,
                                                  len(local_df)).\
                                                  reshape(len(pos_starts_),
                                                          len(local_df))
                        mat_ann_stop = np.repeat(local_df.stop.values,
                                                 len(pos_stops_)).\
                                                 reshape(len(local_df),
                                                         len(pos_stops_)).T
                        mat_int_stop = np.repeat(pos_stops_,
                                                 len(local_df)).\
                                                 reshape(len(pos_stops_),
                                                         len(local_df))
                        if self.seq2seq:
                            _, indexes, _ = np.intersect1d(np.where(mat_ann_start - mat_int_stop < 0)[0],
                                                     np.where(mat_ann_stop - mat_int_start > 0)[0],
                                                     return_indices=True)
                            idx_int, idx_ann = np.where(mat_ann_start - mat_int_stop < 0)
                            idx_int = idx_int[indexes]
                            idx_ann = idx_ann[indexes]
                            
                            offset_start = mat_ann_start - mat_int_start
                            offset_stop = mat_int_stop - mat_ann_stop
        
                            for i, j in zip(idx_int, idx_ann):
                                labels_[i + first_idx,
                                        max(offset_start[i, j], 0) :\
                                        labels_.shape[1] - max(offset_stop[i,j], 0),
                                        cell_type - 1,
                                        label - 1] = 1
        
                        elif not self.seq2seq and self.defined_positive == 'match_all':
                            indexes = np.intersect1d(np.where(mat_ann_start - mat_int_start >= 0)[0],
                                                     np.where(mat_ann_stop - mat_int_stop <= 0)[0])
                            indexes += first_idx
                            labels_[indexes, cell_type - 1, label - 1] = 1
                        elif not self.seq2seq and self.defined_positive == 'match_any':
                            indexes = np.intersect1d(np.where(mat_ann_start - mat_int_stop < 0)[0],
                                                     np.where(mat_ann_stop - mat_int_start > 0)[0])
                            indexes += first_idx
                            labels_[indexes, cell_type - 1, label - 1] = 1

            labels = np.append(labels, labels_, axis=0)

        return labels[1:]

    @property
    def command_dict(self):
        return utils.ArgumentsDict(self, kwargs=False)


class ContinuousDataset(object):
    """
    info:
        docs: >
            Reads files adaptated for continuous annotation (wig, BigWig,
            bedGraph), and returns intervals and the corresponding annotation
            as a label.
            
            An interval can be labeled with two manners. First, the label is
            the experimental values on a window at the center of the interval
            (window of any length within the interval). Secondly, it can be
            labeled by the experimental values covering all the interval and
            downsampled to reach a smaller length. Downsampling can be achived
            by taking one value from several ones or by averaging the values
            within small window.
            Note that wig and bedGraph will be converted and a file with the
            chromosomes size is needed.
    
    args:
        annotation_files:
            list of file with annotations (wig, bigWig or bedGraph).
            If we just want the inputs then a file finishing by .sizes can be
            passed (it must contains the size of chromosome).
         window:
            the length of the intervals.
        tg_window:
            the length of the target window (should be a divisor of the window
            length if downsampling). default=1
        nb_annotation_type:
            The number of different annotation in input files. The same number
            of files must be passed for every annotation.
            The list must be organised as [file1_ann1, file1_ann2, file2_ann1,
            file2_ann2], with file1, file2 designing two differents kind of
            files (different lab, different cellular type ...).
            If None the output shape will be (batch_size, tg_window, nb_of_file)
        downsampling:
            {None, 'mean', 'downsampling'}, how the label is created, if None
            the label is the original values at the center of the interval, if
            'mean' downsampling by averaging on N values recursively,
            if 'downsampling' taking the first value every N values.
            default=None
        normalization_mode:
            arguments from Normalizer class to normalize the data.
            default=None
        overlapping:
            boolean, weither or not to return all the possible intervals, if
            not only the intervals corresponding to non overlapping target will
            be returned. default=True.
        num_chr:
            if specified, 'chr' in the chromosome name will be dropped,
            default=False
        incl_chromosomes:
            exclusive list of chromosome names to include in the final dataset.
            if not None, only these will be present in the dataset,
            default=None
        excl_chromosomes:
            list of chromosome names to omit from the dataset. default=None
        ignore_targets: 
            if True, target variables are ignored, default=False
        size:
            A file with the chromosome name and size usefull to convert wig
            and bedGraph to bigwig.
            default= None
    """
    def __init__(self, annotation_files,
                       window,
                       tg_window=1,
                       nb_annotation_type=None,
                       downsampling=None,
                       normalization_mode=None,
                       overlapping=True,
                       num_chr=False,
                       incl_chromosomes=None,
                       excl_chromosomes=None,
                       ignore_targets=False,
                       size=None):
        
        self.annotation_files = annotation_files
        self.nb_annotation_type = nb_annotation_type
        self.window = window
        self.hw = window // 2
        self.tg_window = tg_window
        self.downsampling = downsampling
        self.normalization_mode = normalization_mode
        self.overlapping = overlapping
        self.num_chr = num_chr
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.ignore_targets = ignore_targets
        self.df = pd.DataFrame()
        self.size = size
        self.frame = inspect.currentframe()

        # converting to list type to consistancy with the case of multi-outputs
        if not isinstance(self.annotation_files, list):
            self.annotation_files = [self.annotation_files]
        self.chrom_size = dict()
        
        for idx, annotation_file in enumerate(self.annotation_files):
            if annotation_file.endswith('.wig'):
                assert self.size is not None,\
                '''To use wig file a file with the chromosome size must be
                parsed a size'''
                utils.wig_to_df(annotation_file, self.size)
                self.annotation_files[idx] = annotation_file[:-3] + 'bw'
            if annotation_file.endswith('.bedGraph'):
                assert self.size is not None,\
                '''To use bedGraph file a file with the chromosome size must be
                parsed a size'''
                utils.bedGraph_to_df(annotation_file, self.size)
                self.annotation_files[idx] = annotation_file[:-8] + 'bw'

        if self.annotation_files[0].endswith(('.wig', '.bw', 'bedGraph')):
            bw = pyBigWig.open(self.annotation_files[0])
            # omit data outside chromosomes
            if incl_chromosomes is not None:
                for name, size in bw.chroms().items():
                    if name in incl_chromosomes:
                        self.chrom_size[name] = size
                        
            elif excl_chromosomes is not None:
                for name, size in bw.chroms().items():
                    if name not in excl_chromosomes:
                        self.chrom_size[name] = size
            else:
                self.chrom_size = bw.chroms()
            bw.close()

        elif self.annotation_files[0].endswith('.sizes'):
            chrom_size = pd.read_csv(self.annotation_files[0],
                                     sep='\t',
                                     names=['chrom', 'sizes'])

            if incl_chromosomes is not None:
                for name, size in zip(chrom_size.chrom.values,
                                      chrom_size.sizes.values):
                    if name in incl_chromosomes:
                        self.chrom_size[name] = size
    
            elif excl_chromosomes is not None:
                for name, size in zip(chrom_size.chrom.values,
                                      chrom_size.sizes.values):
                    if name not in excl_chromosomes:
                        self.chrom_size[name] = size

        self.asteps=1
        if not self.downsampling:
            if not self.overlapping:
                self.asteps = self.tg_window

        else:
            if not self.overlapping:
                self.asteps = self.window

        self.df = self._get_dataframe()

        if not self.ignore_targets:
            self.extractor = bbi_extractor(self.annotation_files,
                                           self.tg_window,
                                           self.nb_annotation_type,
                                           self.downsampling,
                                           self.normalization_mode)

        if self.num_chr and self.df.iloc[0][0].startswith("chr"):
            self.df.chrom = self.df.chrom.str.replace("^chr", "")
        if not self.num_chr and not self.df.iloc[0][0].startswith("chr"):
            self.df.chrom = "chr" + self.df.chrom
    
    def _get_dataframe(self):
        chrom = list()
        start = list()
        stop = list()
        first_index = list()
        last_index = list()
        
        for name, size in self.chrom_size.items():
            chrom.append(name)
            start.append(self.hw)
            stop.append(size - self.hw - (self.window % 2))
            first_index.append(0)
            last_index.append((stop[-1] - start[-1]) // self.asteps)
        
        last_index = np.cumsum(last_index)
        for i in range(len(last_index)):
            last_index[i] += i
        first_index[1:] = last_index[:-1] + 1

        new_df = pd.DataFrame({'chrom' : chrom,
                               'start' : start,
                               'stop' : stop,
                               'first_index' : first_index,
                               'last_index' : last_index})
        return new_df

    def _get_interval(self, idx):
        indicative_mat = (np.sign(self.df.first_index.values - idx)) *\
                         (np.sign(self.df.last_index.values - idx))
        df_idx = np.where(indicative_mat <= 0)[0][-1]

        row = self.df.iloc[df_idx]
        start = row.start + (idx - row.first_index) * self.asteps - self.hw
        stop = row.start + (idx - row.first_index) * self.asteps + self.hw + (self.window % 2)
        interval = pybedtools.create_interval_from_list([row.chrom,
                                                         int(start),
                                                         int(stop)])
        return interval

    def __getitem__(self, idx):
        """Returns (pybedtools.Interval, labels)"""
        if not isinstance(idx, list):
            idx = [idx]

        intervals = [self._get_interval(index) for index in idx]

        if self.ignore_targets:
            labels = {}
        else:
            labels = np.array([self.extractor.extract(interval) for interval in intervals])

        return intervals, labels

    def __len__(self):
        return self.df.last_index.values[-1] + 1

    @property
    def command_dict(self):
        return utils.ArgumentsDict(self, kwargs=False)


class StringSeqIntervalDl(Dataset):
    """
    info:
        docs: >
            Dataloader for a combination of fasta and a file with annotations.
            The dataloader extracts regions from the fasta file corresponding
            to the `annotation_file`. Returned sequences are of the type
            np.array([str]), possibly the corresponding occupancy taken from a
            bbi file can be passed as secondary input or targets.
    args:
        annotation_files:
            list of file with annotations (wig, bigWig or bedGraph / bed, gff)
        fasta_file:
            Reference genome FASTA file path.
        force_upper:
            Force uppercase output of sequences
        use_strand:
            boolean, whether or not to respect the strand for spare annotation.
            If false all the sequence are ridden in 5'.
        sec_inputs:
            Path to other bbi files, the corresponding coverage on the interval
            will be used as a secondary input for the model (or targets)
            default=None
        sec_input_length:
            {int, 'maxlen'}, Length of the secondary sequences to be used as
            model input. If maxlen the length will be the same as the DNA seq.
            default='maxlen'
        sec_input_shape:
            The shape of the secondary inputs, in order to adapt this shape to
            the need.
            default=None.
        sec_nb_annotation:
            The number of different annotation in secondary input files.
            (see ContinuousDataset for details).
            default=None
        sec_sampling_mode:
            How the secondary inputs are sampled from the coverage on the cor-
            -responding input interval.
            default=None
        sec_normalization_mode:
            How the secondary inputs are normalized.
        use_sec_as:
            {'inputs', 'targets'}
            default='inputs'
        rc:
            boolean, if true the batch is reversed complemented.
            default=False
        args: 
            Arguments to be passed to the dataset reader
        kwargs: 
           Dictionnary of arguments specific to the dataset reader
    output_schema:
        inputs:
            name: seq
            shape: ()
            doc: DNA sequence as string
            special_type: DNAStringSeq
            associated_metadata: ranges
        targets:
            shape: (None,)
            doc: (optional) values corresponding to the annotation file
        metadata:
            ranges:
                type: GenomicRanges
                doc: Ranges describing inputs.seq
    """
    def __init__(self,
                 annotation_files,
                 fasta_file,
                 use_strand=False,
                 sec_inputs=None,
                 sec_input_length='maxlen',
                 sec_input_shape=None,
                 sec_nb_annotation=None,
                 sec_sampling_mode=None,
                 sec_normalization_mode=None,
                 use_sec_as='inputs',
                 force_upper=False,
                 rc=False,
                 *args,
                 **kwargs):
        self.annotation_files = annotation_files
        self.fasta_file = fasta_file
        self.use_strand = use_strand
        self.force_upper = force_upper
        self.fasta_extractors = None
        self.pad_seq = False
        self.sec_inputs = sec_inputs
        self.sec_input_length = sec_input_length
        self.sec_input_shape = sec_input_shape
        self.sec_nb_annotation = sec_nb_annotation
        self.sec_sampling_mode = sec_sampling_mode
        self.sec_normalization_mode = sec_normalization_mode
        self.use_sec_as = use_sec_as
        self.rc = rc
        self.frame = inspect.currentframe()

        assert self.use_sec_as in ['targets', 'inputs'],\
        'use_sec_as is either "targets" or "input"'

        if not isinstance(self.annotation_files, list):
            self.annotation_files = [self.annotation_files]

        if self.annotation_files[0].endswith(('.bed', '.gff', 'gff3', 'gtf')):
           self.dataset = SparseDataset(annotation_files = self.annotation_files,
                                        *args,
                                        **kwargs)

           if self.dataset.seq_len == 'real':
               self.pad_seq = True

        elif self.annotation_files[0].endswith(('.wig', '.bw', 'bedGraph', '.sizes')):
            self.dataset = ContinuousDataset(annotation_files = self.annotation_files,
                                             *args,
                                             **kwargs)
        if self.sec_input_length == 'maxlen':
            try:
                self.sec_input_length = self.dataset.length
            except AttributeError:
                self.sec_input_length = self.dataset.window

        if self.sec_inputs:
            self.extractor = bbi_extractor(self.sec_inputs,
                                           self.sec_input_length,
                                           self.sec_nb_annotation,
                                           self.sec_sampling_mode,
                                           self.sec_normalization_mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not isinstance(idx, list):
            idx = [idx]

        self.fasta_extractors = FastaStringExtractor(self.fasta_file,
                                                     use_strand=self.use_strand,
                                                     force_upper=self.force_upper)

        intervals, labels = self.dataset[idx]
        seqs = list()

        if self.use_strand:
            negative_strand = list()

            assert hasattr(intervals[0], 'strand'),\
            '''Strand need to be specified to use use_strand'''

            for i in range(len(intervals)):
                interval = intervals[i]
                if interval.length == 0:
                    seqs.append(''.join(random.choices('ATGC',
                                                       k=self.dataset.length)))
                elif interval.strand == '-':
                    seqs.append(self.fasta_extractors.extract(interval))
                    negative_strand.append(i)
                else:
                    seqs.append(self.fasta_extractors.extract(interval))

            if self.dataset.seq2seq == True:
                labels[negative_strand] = labels[negative_strand, ::-1, :, :]

        else:  
            for interval in intervals:
                if interval.length == 0:
                    seqs.append(''.join(random.choices('ATGC',
                                                       k=self.dataset.length)))
                else:
                    seqs.append(self.fasta_extractors.extract(interval) )

        if self.pad_seq:
                seqs = [fixed_len(seq,
                             int(self.dataset.length),
                             anchor="center",
                             value="N") for seq in seqs]

        if self.sec_inputs:
            sec_seqs = [self.extractor.extract(interval) for interval in intervals]
            sec_seqs = np.array(sec_seqs)
            
            if self.use_strand:
                sec_seqs[negative_strand] = sec_seqs[negative_strand, ::-1]            

            if self.rc:
                seqs, sec_seqs, labels = utils.reverse_complement(seqs,
                                                                 labels,
                                                                 sec_seqs)
            if self.sec_input_shape:
                sec_seqs = sec_seqs.reshape((sec_seqs.shape[0],) + \
                                             self.sec_input_shape[1:])
            if self.use_sec_as == 'inputs':
                inputs = [np.array(seqs), sec_seqs]
            else:
                inputs = np.array(seqs)
                labels = [labels, sec_seqs]
        else:
            if self.rc:
                seqs, labels = utils.reverse_complement(seqs, labels)
            inputs = np.array(seqs)

        return {
            "inputs": inputs,
            "targets": labels,
            "metadata": {
                "ranges": [GenomicRanges(interval.chrom,
                                         interval.start,
                                         interval.stop,
                                         str(idx_)) for interval, idx_ in zip(intervals, idx)] 
            }
        }

    @property
    def command_dict(self):
        return utils.ArgumentsDict(self, called_args='dataset')


class SeqIntervalDl(Dataset):
    """
    info:
        doc: >
            Dataloader for a combination of fasta and tab-delimited input files
            such as bed files. The dataloader extracts regions from the fasta
            file corresponding to the `annotation_file` and converts them into
            one-hot encoded format. Returned sequences are of the type np.array
            with the shape inferred from the arguments: `alphabet_axis` and
            `dummy_axis`.
    args:
        alphabet_axis:
            doc: axis along which the alphabet runs (e.g. A,C,G,T for DNA)
        dummy_axis:
            doc: defines in which dimension a dummy axis should be added.
            None if no dummy axis is required.
        alphabet:
            doc: >
                alphabet to use for the one-hot encoding. This defines the
                order of the one-hot encoding.
                Can either be a list or a string: 'ACGT' or ['A, 'C', 'G', 'T'].
                Default: 'ACGT'
        dtype:
            doc: 'defines the numpy dtype of the returned array.
            Example: int, np.int32, np.float32, float'
        args: 
            arguments specific to the different dataloader that can be used
        kwargs: 
            dictionnary with specific arguments to the dataloader.
    output_schema:
        inputs:
            name: seq
            shape: (None, 4)
            doc: One-hot encoded DNA sequence
            special_type: DNASeq
            associated_metadata: ranges
        targets:
            shape: (None,)
            doc: (optional) values given in the annotation file
        metadata:
            ranges:
                type: GenomicRanges
                doc: Ranges describing inputs.seq
    """
    def __init__(self,
                 alphabet_axis=1,
                 dummy_axis=None,
                 alphabet=DNA,
                 dtype=None,
                 *args,
                 **kwargs):
        self.frame = inspect.currentframe()
        # core dataset, not using the one-hot encoding params
        self.seq_dl = StringSeqIntervalDl(*args,
                                          **kwargs)

        self.input_transform = ReorderedOneHot(alphabet=alphabet,
                                               dtype=dtype,
                                               alphabet_axis=alphabet_axis,
                                               dummy_axis=dummy_axis)

    def __len__(self):
        return len(self.seq_dl)

    def __getitem__(self, idx):
        ret = self.seq_dl[idx]
        
        if self.seq_dl.sec_inputs and self.seq_dl.use_sec_as == 'inputs':
            length = len(ret['inputs'][0])
            ret['inputs'] = [np.array([self.input_transform(str(ret["inputs"][0][i]))\
                             for i in range(length)]), ret['inputs'][1]]
        else:   
            length = len(ret['inputs'])
            ret['inputs'] = np.array([self.input_transform(str(ret["inputs"][i]))\
                            for i in range(length)])
        return ret

    @property
    def command_dict(self):
        return utils.ArgumentsDict(self, called_args='seq_dl')
    