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
import sys


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
            {'real', 'random', bedfile, None} if real the negative example will be
            taken from sequences far enough from any annotation example. If None this
            function will return only positive example, 'random' will return 
            interval of length 0. Pass a bedfile to precisely set the negative examples.
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

        if self.seq2seq:
            self.defined_positive = 'match_any'

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

        elif not self.negative_type:
            pass

        elif self.negative_type.endswith('.bed'):
            neg_df, neg_label = self._negative_bed()
            self.df = self.df.append(neg_df)

            if not self.ignore_targets:
                self.labels = np.append(self.labels, neg_label, axis=0)

    @classmethod
    def default_dict(cls):
        return utils.get_default_args(cls.__init__)

    @classmethod
    def predict_label_shape(cls, **input_dict):
        command_dict = cls.default_dict()
        command_dict.update(input_dict)

        if command_dict['ignore_targets']:
            return None

        assert 'annotation_files' in command_dict,\
        """To create an instance passing annotation_files is required"""

        assert 'annotation_list' in command_dict,\
        """To create an instance passing annotation_list is required"""

        if command_dict['seq2seq']:
            assert not isinstance(command_dict['seq_len'], str),\
            """The label shape can only anticipated if seq_len is an integer"""

        if isinstance(command_dict['annotation_files'], str):
            command_dict['annotation_files'] = [command_dict['annotation_files']]
        
        if command_dict['seq2seq']:
            return (command_dict['seq_len'],
                    len(command_dict['annotation_files']),
                    len(command_dict['annotation_list']))
        else:
            return (len(command_dict['annotation_files']),
                    len(command_dict['annotation_list']))

    @property
    def label_shape(self):
        command_dict = self.command_dict.as_input()
        command_dict['seq_len'] = self.length
        return self.predict_label_shape(**command_dict)

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
        if return_strand:
            assert 'strand' in df.columns, \
            'To return the strand the dataframe should have a strand column'

        start = df.start.values
        stop = df.stop.values

        if self.data_augmentation and return_all:
            starts = np.concatenate([np.arange(stop[i] - self.length,
                                               start[i] + 1)\
                                     if start[i] - stop[i] + self.length >= 0\
                                     else np.arange(start[i], stop[i] - self.length)\
                                     for i in range(len(start))],
                                     axis=0)
            stops = np.concatenate([np.arange(stop[i],
                                              start[i] + 1 + self.length)\
                                    if start[i] - stop[i] + self.length >= 0\
                                    else np.arange(start[i] + self.length, stop[i])\
                                    for i in range(len(start))],
                                    axis=0)

            if return_strand:
                strand = df.strand.values
                strands = np.concatenate([np.repeat(strand[i],
                                                    start[i] - stop[i] + self.length + 1)\
                                          if stop[i] - start[i] <= self.length
                                          else np.repeat(strand[i],
                                                         stop[i] - start[i] - self.length)
                                          for i in range(len(strand))\
                                           ])
                return starts, stops, strands
            else:
                return starts, stops

        elif self.data_augmentation and not return_all:
            starts = stop - self.length
            stops = start + self.length

            mask = start + self.length - stop <= 0
            starts[mask] = start[mask]
            stops[mask] = stop[mask]
            return starts, stops 

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
    
        for chrom in self.ann_df.chrom.unique():
            df_ = self.ann_df[self.ann_df.chrom == chrom]
            df_ = df_.sort_values('start')
            pos_starts, pos_stops = self._calculate_interval(df_)
            number_of_pos = np.sum(pos_stops - pos_starts)

            pos_starts = pos_starts - self.length
            list_interval = np.where((pos_starts[1:] - pos_stops[:-1]) > 0)[0]

            if self.negative_ratio == 'all':
                neg_starts = np.concatenate([np.arange(pos_stops[interval_idx],
                                                       pos_starts[interval_idx + 1])\
                                             for interval_idx in list_interval],
                                            0)

            elif isinstance(self.negative_ratio, int):
                length_inters = np.array([pos_starts[i + 1] - pos_stops[i] for i in list_interval])
                proba = length_inters / np.sum(length_inters)

                if self.data_augmentation:
                    number_of_pos *= self.negative_ratio
                else:
                    number_of_pos = len(self.ann_df[self.ann_df.chrom == chrom])
                    number_of_pos *= self.negative_ratio

                interval_chosen = np.random.choice(list_interval,
                                                   number_of_pos,
                                                   p=proba)

                nb_per_interval = [min((interval_chosen == inter).sum(), length_inter)\
                                   for inter, length_inter in zip(list_interval, length_inters)]

                neg_starts = np.concatenate([np.random.choice(np.arange(pos_stops[interval_idx],
                                                                        pos_starts[interval_idx + 1]),
                                                              nb_inter,
                                                              replace=False) for interval_idx,\
                                                                                 nb_inter in\
                                                                                 zip(list_interval,
                                                                                     nb_per_interval)],
                                            0)
            else:
                raise NameError('negative_ratio should be "all" or an integer')

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
    
    def _negative_bed(self):
        neg_df = utils.bed_to_df(self.negative_type, [0])
        incl_chromosomes = self.ann_df.chrom.unique()
        neg_df = neg_df[neg_df.chrom.isin(incl_chromosomes)]

        assert (neg_df.stop.values - neg_df.start.values == self.length).all(),\
        """The negative examples must be of the same length as the positive one"""
        
        neg_df['type'] = 0
        
        if hasattr(self.ann_df, 'strand'):
            assert hasattr(neg_df, 'strand'),\
            """Strand is specifies for positive class and must be specified for
            the negative class as well"""
        else:
            assert not hasattr(neg_df, 'strand'),\
            """Strand is NOT specifies for positive class and must NOT be specified
            for the negative class neither"""
        nb_types = len(self.ann_df.type.unique())
        nb_labels = len(self.ann_df.label.unique())

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

    def _get_translation_dico(self):
        trans_dico = {}

        for chrom in self.ann_df.chrom.unique():
            df_ = self.ann_df[self.ann_df.chrom == chrom]
            starts = df_.start.values
            stops = df_.stop.values

            if self.data_augmentation:
                len_per_ann = [max(stops[i] - starts[i] - self.length,
                                   starts[i] - stops[i] + self.length + 1) for i in range(len(starts))]
                base_index = np.cumsum(len_per_ann)
                trans_dico[chrom] = {i : np.arange(base_index[i - 1], base_index[i]) \
                                     for i in range(len(starts))}

                trans_dico[chrom][0] = np.arange(base_index[0])
            else:
                trans_dico[chrom] = {i : [i] for i in range(len(starts))}
        return trans_dico

    def _annotation_inter(self):
        inter_dict = {}

        for chrom in self.ann_df.chrom.unique():
            local_df = self.ann_df[self.ann_df.chrom == chrom]
            starts = local_df.start.values
            stops = local_df.stop.values
            mat_ann_start = np.repeat(starts,
                                      len(starts)).reshape((len(starts),
                                                            len(starts)))
            mat_ann_stop = np.repeat(stops,
                                     len(stops)).reshape((len(starts),
                                                          len(starts)))

            if self.defined_positive == 'match_all':
                A = np.minimum(np.abs(mat_ann_stop - mat_ann_stop.T),
                               np.abs(mat_ann_start - mat_ann_start.T))
                m = A.shape[0]
                strided = np.lib.stride_tricks.as_strided
                s0, s1 = A.strides
                out = strided(A.ravel()[1:],
                              shape=(m - 1, m),
                              strides=(s0 + s1, s1)).reshape(m, -1)
            else:
                A = np.minimum(np.abs(mat_ann_stop - mat_ann_stop.T),
                               np.abs(mat_ann_start - mat_ann_start.T),
                               np.abs(mat_ann_start - mat_ann_stop.T))
                m = A.shape[0]
                strided = np.lib.stride_tricks.as_strided
                s0, s1 = A.strides
                out = strided(A.ravel()[1:],
                              shape=(m - 1, m),
                              strides=(s0 + s1, s1)).reshape(m, -1)
            if self.data_augmentation:
                indexes = np.where(out <= self.length)[0]
            else:
                indexes = np.where(out <= self.length // 2)[0]

            inter_indexes = np.unique(indexes)
            inter_dict[chrom] = [np.delete(np.arange(len(starts)), inter_indexes),
                                 inter_indexes]
        return inter_dict

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
        trans_dict = self._get_translation_dico()
        inter_dict = self._annotation_inter()

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
            local_df = self.df[self.df.chrom == chrom]
            ann_to_df = trans_dict[chrom]
            no_inter, inter = inter_dict[chrom]

            if self.seq2seq:
                seq_starts = {idx :
                              np.maximum(np.zeros(len(ann_to_df[idx])).astype(int), 
                                     df_.start.values[idx] - local_df.start.values[ann_to_df[idx]])\
                              for idx in no_inter}
                seq_stops = {idx :
                             np.minimum(np.ones(len(ann_to_df[idx])).astype(int) * self.length,
                                    df_.stop.values[idx] - local_df.start.values[ann_to_df[idx]])\
                             for idx in no_inter}

                labels_[np.concatenate([np.concatenate([np.repeat(ann_to_df[idx][i],
                                                                  seq_stops[idx][i] - seq_starts[idx][i])\
                                                       for i in range(len(ann_to_df[idx]))], 0)\
                                        for idx in no_inter]),
                        np.concatenate([np.concatenate([np.arange(seq_starts[idx][i],
                                                                  seq_stops[idx][i])\
                                                       for i in range(len(ann_to_df[idx]))], 0)\
                                        for idx in no_inter]),
                        np.concatenate([np.repeat(df_.type.values[idx] - 1,
                                                  len(ann_to_df[idx]) *\
                                                  min(self.length,
                                                      df_.stop.values[idx] - df_.start.values[idx]))\
                                        for idx in no_inter]),
                        np.concatenate([np.repeat(df_.label.values[idx] - 1,
                                                  len(ann_to_df[idx]) *\
                                                  min(self.length,
                                                      df_.stop.values[idx] - df_.start.values[idx]))\
                                        for idx in no_inter])] = 1
            else:
                labels_[np.concatenate([ann_to_df[idx] for idx in no_inter], 0),
                        np.concatenate([np.repeat(df_.type.values[idx] - 1,
                                                  len(ann_to_df[idx])) for idx in no_inter], 0),
                        np.concatenate([np.repeat(df_.label.values[idx] - 1,
                                                  len(ann_to_df[idx])) for idx in no_inter], 0)] = 1
            if len(inter) > 0:
                df_inter = np.concatenate([ann_to_df[idx] for idx in inter])
                pos_starts = local_df.start.values[df_inter]
                pos_stops = local_df.stop.values[df_inter]

                ann_starts = df_.start.values[inter]
                ann_stops = df_.stop.values[inter]

                mat_ann_start = np.repeat(ann_starts,
                                          len(pos_starts)).\
                                          reshape(len(ann_starts),
                                          len(pos_starts)).T
                mat_int_start = np.repeat(pos_starts,
                                          len(ann_starts)).\
                                          reshape(len(pos_starts),
                                          len(ann_starts))

                mat_ann_stop = np.repeat(ann_stops,
                                         len(pos_stops)).\
                                         reshape(len(ann_stops),
                                         len(pos_stops)).T
                mat_int_stop = np.repeat(pos_stops,
                                         len(ann_stops)).\
                                         reshape(len(pos_stops),
                                         len(ann_stops))

                if self.seq2seq:
                    idx_int, idx_ann = np.where(np.sign(mat_ann_start - mat_int_stop) *\
                                             np.sign(mat_ann_stop - mat_int_start) < 0)
                    offset_start = mat_ann_start - mat_int_start
                    offset_start[offset_start < 0] = 0 
                    offset_stop = self.length - mat_int_stop + mat_ann_stop
                    offset_stop[offset_stop < 0] = 0

                    for i, j in zip(idx_int, idx_ann):
                        labels_[df_inter[i],
                                offset_start[i, j] : offset_stop[i, j],
                                df_.type.values[inter[j]] - 1,
                                df_.label.values.astype(int)[inter[j]] - 1] = 1

                else:
                    if self.defined_positive == 'match_all':
                        idx_int, idx_ann = np.where(np.sign(mat_ann_start - mat_int_start) *\
                                                    np.sign(mat_ann_stop - mat_int_stop) <= 0)
                    else:
                        idx_int, idx_ann = np.where(np.sign(mat_ann_start - mat_int_stop) *\
                                             np.sign(mat_ann_stop - mat_int_start) < 0)

                    labels_[df_inter[idx_int],
                            df_.type.values[inter[idx_ann]] - 1,
                            df_.label.values[inter[idx_ann]].astype(int) - 1] = 1
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
            boolean or int, weither or not to return all the possible intervals, if
            False only the intervals corresponding to non overlapping target will
            be returned, if int the stride between the interval will be of int.
            default=True.
        num_chr:
            if specified, 'chr' in the chromosome name will be dropped,
            default=False
        incl_chromosomes:
            exclusive list of chromosome names to include in the final dataset.
            if not None, only these will be present in the dataset,
            default=None
        excl_chromosomes:
            list of chromosome names to omit from the dataset. default=None
        start_stop:
            list of tuple indicating where to start and stop generating data.
            One per included chromosome. default=None
        ignore_targets: 
            if True, target variables are ignored, default=False
        size:
            A file with the chromosome name and size usefull to convert wig
            and bedGraph to bigwig.
            default= None
        path_to_ucsc:
            The path to the ucsc utils (wigToBigWig and bedGraphToBigWig) to
            give if those functionality can not be downloaded with conda.
            default=None
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
                       start_stop=None,
                       ignore_targets=False,
                       size=None,
                       path_to_ucsc=None):
        
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
        self.start_stop = start_stop
        self.ignore_targets = ignore_targets
        self.df = pd.DataFrame()
        self.size = size
        self.path_to_ucsc = path_to_ucsc
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
                utils.wig_to_df(annotation_file,
                                self.size,
                                self.path_to_ucsc)
                self.annotation_files[idx] = annotation_file[:-3] + 'bw'
            if annotation_file.endswith('.bedGraph'):
                assert self.size is not None,\
                '''To use bedGraph file a file with the chromosome size must be
                parsed a size'''
                utils.bedGraph_to_df(annotation_file,
                                     self.size,
                                     self.path_to_ucsc)
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
        
        if self.start_stop is not None:
            if isinstance(self.start_stop, tuple):
                self.start_stop = [self.start_stop]
            assert len(self.start_stop) == len(self.chrom_size),\
            """len(start_stop) should match the number of chromosome
            included. Found {} and expected {}""".format(len(self.start_stop),
                                                         len(self.chrom_size))

            for size, start_stop in zip(self.chrom_size.values(), self.start_stop):
                assert start_stop[0] >= self.hw,\
                """start in start_stop must be greater than the half of window"""
                assert start_stop[1] < size - self.hw,\
                """stop in start_stop must be smaller than the chromosome size
                minus the half of window"""

        self.asteps = 1
        if isinstance(self.overlapping, int):
            self.asteps = self.overlapping

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

    @classmethod
    def default_dict(cls):
        return utils.get_default_args(cls.__init__)

    @classmethod
    def predict_label_shape(cls, **input_dict):
        command_dict = cls.default_dict()
        command_dict.update(input_dict)

        if command_dict['ignore_targets']:
            return None

        assert 'annotation_files' in command_dict,\
        """To create an instance passing annotation_files is required"""

        assert 'window' in command_dict,\
        """To create an instance passing window is required"""

        if isinstance(command_dict['annotation_files'], str):
            command_dict['annotation_files'] = [command_dict['annotation_files']]
        
        if isinstance(command_dict['nb_annotation_type'], int):
            assert len(command_dict['annotation_files'])\
            % command_dict['nb_annotation_type'] == 0,\
            """nb_annotation_type should devide the number of annotation files"""
            return (command_dict['tg_window'],
                    len(command_dict['annotation_files']) // command_dict['nb_annotation_type'],
                    command_dict['nb_annotation_type'])
        else:
            return (command_dict['tg_window'],
                    len(command_dict['annotation_files']))

    @property
    def label_shape(self):
        command_dict = self.command_dict.as_input()
        return self.predict_label_shape(**command_dict)

    def _get_dataframe(self):
        chrom = list()
        start = list()
        stop = list()
        first_index = list()
        last_index = list()

        if self.start_stop is not None:
            for name, start_stop in zip(self.incl_chromosomes,
                                        self.start_stop):
                chrom.append(name)
                start.append(start_stop[0])
                stop.append(start_stop[1])
                first_index.append(0)
                last_index.append((stop[-1] - start[-1]) // self.asteps)
        else:
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


class StringSeqIntervalDl(object):
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

    @classmethod
    def default_dict(cls):
        return utils.get_default_args(cls.__init__)

    @classmethod
    def predict_sec_input_shape(cls, **input_dict):
        command_dict = cls.default_dict()
        command_dict.update(input_dict)

        if command_dict['sec_inputs'] is None:
            return None

        assert not isinstance(command_dict['sec_input_length'], str),\
        """To anticipate the secondary shape sec_input_length must be an integer"""

        if isinstance(command_dict['sec_inputs'], str):
            command_dict['sec_inputs'] = [command_dict['sec_inputs']]

        if isinstance(command_dict['sec_nb_annotation'], int):
            assert len(command_dict['sec_inputs'])\
            % command_dict['sec_nb_annotation'] == 0,\
            """sec_nb_annotation should devide the length of sec_inputs"""
            
            sec_input_shape = (command_dict['sec_input_length'],
                               len(command_dict['sec_inputs'])\
                               // command_dict['sec_nb_annotation'],
                               command_dict['sec_nb_annotation'])
        else:
            sec_input_shape =  (command_dict['sec_input_length'],
                                len(command_dict['sec_inputs']))
            
        if command_dict['sec_input_shape']:
            try:
                np.zeros(sec_input_shape).reshape(command_dict['sec_input_shape'][1:])
                return command_dict['sec_input_shape'][1:]
            except ValueError:
                raise ValueError("""The required secondary input shape is not compatible with the 
                other arguments""")
                
        else:
            return sec_input_shape

    @property
    def secondary_input_shape(self):
        command_dict = self.command_dict.as_input()
        command_dict['sec_input_length'] = self.sec_input_length
        return self.predict_sec_input_shape(**command_dict)

    @classmethod
    def predict_label_shape(cls, **input_dict):
        assert 'annotation_files' in input_dict,\
        """To create an instance passing annotation_files is required"""

        if isinstance(input_dict['annotation_files'], str):
            input_dict['annotation_files'] = [input_dict['annotation_files']]
        
        if input_dict['annotation_files'][0].endswith(('.bed', '.gff', 'gff3', 'gtf')):
            return SparseDataset.predict_label_shape(**input_dict)
        elif input_dict['annotation_files'][0].endswith(('.wig', '.bw', 'bedGraph')):
            return ContinuousDataset.predict_label_shape(**input_dict)

    @property
    def label_shape(self):
        return self.dataset.label_shape

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
            }

    @property
    def command_dict(self):
        return utils.ArgumentsDict(self, called_args='dataset')


class SeqIntervalDl(object):
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

    @classmethod
    def default_dict(cls):
        return utils.get_default_args(cls.__init__)
    
    @classmethod
    def predict_sec_input_shape(cls, **input_dict):
        return StringSeqIntervalDl.predict_sec_input_shape(**input_dict)

    @property
    def secondary_input_shape(self):
        return self.seq_dl.secondary_input_shape
    
    @classmethod
    def predict_label_shape(cls, **input_dict):
        return StringSeqIntervalDl.predict_label_shape(**input_dict)
    
    @property
    def label_shape(self):
        command_dict = self.command_dict.as_input()
        return self.seq_dl.label_shape
    
    @classmethod
    def predict_input_shape(cls, **input_dict):
        command_dict = cls.default_dict()
        command_dict.update(input_dict)
        
        assert "annotation_files" in command_dict,\
        """annotation_files is needed to calculate the input shape"""

        if isinstance(command_dict['annotation_files'], str):
            command_dict['annotation_files'] = [command_dict['annotation_files']]

        if command_dict['annotation_files'][0].endswith(('.bed', '.gff', 'gff3', 'gtf')):
            assert 'seq_len' in command_dict,\
            """seq_len can not be set as default if we want to anticipate the input shape"""
            assert not isinstance(command_dict['seq_len'], str),\
            """seq_len must be an integer to calculate the input shape"""
            length = command_dict['seq_len']
            
        elif command_dict['annotation_files'][0].endswith(('.wig', '.bw', 'bedGraph')):
            assert 'window' in command_dict,\
            """window is needed to calculate the input shape with bigwig files"""
            length = command_dict['window']
            
        if command_dict['dummy_axis']:
            shape = np.zeros((3,), dtype=int)
            shape[command_dict['dummy_axis']] = 1
            shape[command_dict['alphabet_axis']] = 4
            shape[shape == 0] = length
            return tuple(shape)
        else:
            shape = np.zeros((2,), dtype=int)
            shape[command_dict['alphabet_axis']] = 4
            shape[shape == 0] = length
            return tuple(shape)
    
    @property
    def input_shape(self):
        command_dict = self.command_dict.as_input()

        if command_dict['annotation_files'][0].endswith(('.bed', '.gff', 'gff3', 'gtf')):
            command_dict['seq_len'] = self.seq_dl.dataset.length

        return self.predict_input_shape(**command_dict)
    