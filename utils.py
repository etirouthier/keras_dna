#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:30:24 2019

@author: routhier
"""
import numpy as np
import pybedtools
import pandas as pd
import pyBigWig
import os
import inspect

def continuous_weights(x, threshold=10**4, include_zeros=False,
						  correction='max_relative'):
	"""
		Take an array as input and return an array of the same length
		with the weights to be used in the training procedure. The
		dataset will be balanced (the weights are 1 / frequency).

		args:
			x: array with integer values (as usual in genomics)
			threshold: over this threshold the weights is set to 0.
			include_zeros: if false weights corresponding to zeros is 0.
			correction: 'balanced' or 'max_relative'
	"""
	indexes, counts = np.unique(x, return_counts=True)
	weights = np.ones(x.shape)

	if correction == 'max_relative':
		# we calculate the maximum counts without zeros (overrepresented)
		maximum = np.max(counts[1:])

		for index, count in zip(indexes, counts):
			weights[x == index] = maximum / float(count)

	if correction == 'balanced':
		# we calculate the total counts without zeros (overrepresented)
		total = np.sum(counts[1:])

		for index, count in zip(indexes, counts):
			weights[x == index] = total / float(count)

	if not include_zeros:
		weights[x == 0] = 0

	weights[weights > threshold] = 0
	return weights

def bed_to_df(bedfile, annotation_list):
    bedfile = pybedtools.BedTool(bedfile)
    
    assert len(annotation_list) == 1, \
    """A .bed file can only display the position for one type of 
    annotation."""
    
    chroms, starts, stops, labels = [], [], [], []
    if hasattr(bedfile[0], 'strand'):
        strands = []
    
    for bed in bedfile:
        chroms.append(bed.chrom)
        starts.append(bed.start)
        stops.append(bed.stop)
        labels.append(annotation_list[0])
        
        if hasattr(bed, 'strand'):
            strands.append(bed.strand)
    
    if strands:
        df = pd.DataFrame({'chrom': chroms,
                           'start': starts,
                           'stop': stops,
                           'label': labels,
                           'strand': strands})
    else:
        df = pd.DataFrame({'chrom': chroms,
                           'start': starts,
                           'stop': stops,
                           'label': labels})
    return df

def gff_to_df(gff_file, annotation_list):
    gff_file = pybedtools.BedTool(gff_file)
    
    chroms, starts, stops, labels = [], [], [], []
    if not gff_file[0].strand == '.':
        strands = []
    else:
        strands=None
    
    gff_file = gff_file.filter(lambda x : x.fields[2] in annotation_list)
    
    for gff in gff_file:
        chroms.append(gff.chrom)
        starts.append(gff.start)
        stops.append(gff.stop)
        labels.append(gff.fields[2])
        
        if isinstance(strands, list):
            strands.append(gff.strand)
    
    if isinstance(strands, list):
        df = pd.DataFrame({'chrom': chroms,
                           'start': starts,
                           'stop': stops,
                           'label': labels,
                           'strand': strands})
    else:
        df = pd.DataFrame({'chrom': chroms,
                           'start': starts,
                           'stop': stops,
                           'label': labels})
    return df

def bedGraph_to_df(bedGraph, chrom_size):
    os.system('ucsc-utils/bedGraphToBigWig ' + bedGraph + \
              ' ' + chrom_size + ' ' + 'tmp.bw')
    df = bigwig_to_df('tmp.bw')
    
    os.remove('tmp.bw')
    return df

def wig_to_df(wig, chrom_size):
    os.system('ucsc-utils/wigToBigWig ' + wig + \
              ' ' + chrom_size + ' ' + 'tmp.bw')
    df = bigwig_to_df('tmp.bw')
    
    os.remove('tmp.bw')
    return df

def bigwig_to_df(bigfile):
    bw = pyBigWig.open(bigfile)
    df = pd.DataFrame()
    
    for name, length in bw.chroms().items():
        df_ = pd.DataFrame()
        label = np.array(bw.values(name, 0, length))
        label[np.isnan(label)] = 0
        df_['label'] = label
        df_['pos_on_chrom'] = np.arange(length)
        df_['chrom'] = name
        
        df = df.append(df_)
    
    return df

def bbi_extractor(interval, bbi_files, final_dummy_axis=False):
    """
        Function used to extract the coverage of the given bbi file at the
        position of the interval.
    """   
    if not isinstance(bbi_files, list):
        bbi_files = [bbi_files]
    
    seq = list()
    for bbi_file in bbi_files:
        bw = pyBigWig.open(bbi_file)
        seq.append(np.array(bw.values(interval.chrom,
                                      interval.start,
                                      interval.stop)))
    seq =  np.array(seq).T
    
    if final_dummy_axis:
        seq = seq.reshape((seq.shape[0],
                           seq.shape[1],
                           1))
    return seq

def reverse_complement_fa(seq):
    seq = seq[::-1]
    seq = seq.replace('A', '1')
    seq = seq.replace('C', '2')
    seq = seq.replace('T', 'A')
    seq = seq.replace('G', 'C')
    seq = seq.replace('1', 'T')
    seq = seq.replace('2', 'G')
    
    return seq

def reverse_complement(seqs, labels, bbi_seqs=None):
    for i in range(len(seqs)):
        seqs[i] = reverse_complement_fa(seqs[i])
    labels = labels[:, ::-1]
    
    if bbi_seqs is not None:
        bbi_seqs = bbi_seqs[:, ::-1]
    
        return seqs, bbi_seqs, labels
    else:
        return seqs, labels


class ArgumentsDict(object):
    """
    Return dictionnary with the arguments passed to the several class necessary
    to build a generator.
    
    args:
        instance:
            An instance of a class used in the process of building a generator.
        called_args:
            When the instance calls another class able to return an ArgumentsDict,
            called_args is the corresponding argument.
        kwargs:
            If the instance accept kwargs to be passed.
    returns:
        An ArgumentsDict instance with the arguments passed to the class to
        construct a generator.
    """
    def __init__(self,
                 instance,
                 called_args=None,
                 kwargs=True):
        assert hasattr(instance, 'frame'),\
        """To access the arguments dict the class must give access to the
        current frame in the __init__ function with the arguments 'frame' """
        self.frame = instance.frame
        self.instance = instance
        
        if called_args:
            assert hasattr(self.instance, called_args),\
            "The instance must own the arguments {}".format(called_args)
            args_instance = getattr(self.instance, called_args)
            assert hasattr(args_instance, 'command_dict'),\
            """The class called with called_args must be able to return an
            ArgumentsDict instance with 'command_dict'"""
            
        self.called_args = called_args
        self.kwargs = kwargs
    
    def __call__(self):
        """
        Returns a dictionnary with the arguments passed to the instance
        separated in two fields, the arguments corresponding to the class and
        kwargs given to other object used by the instance.
        """
        args, _, _, values = inspect.getargvalues(self.frame)
        dico = {i : values[i] for i in args[1:]}            
        
        if self.kwargs:
            return {'args' : dico, 'kwargs' : values['kwargs']}
        else:
            return dico
        
    def as_input(self):
        """
        Returns a dictionnary with the arguments passed to construct the instan
        ce. Can be used as input to reconstruct the same instance.
        """
        args, _, _, values = inspect.getargvalues(self.frame)
        dico = {i : values[i] for i in args[1:]}            
        
        if self.kwargs:
            dico.update(values['kwargs'])
        return dico
    
    def get_details(self):
        """
        Detailed version of the dictionnary with all the class used in the proc
        ess and their arguments.
        """
        if self.called_args:
            args_instance = getattr(self.instance, self.called_args)
            args_dico = args_instance.command_dict.get_details()
            args_dico.update({str(type(self.instance))[8 : -2] : self.__call__()['args']})
            return args_dico
        else:
            return {str(type(self.instance))[8 : -2] : self.as_input()}
