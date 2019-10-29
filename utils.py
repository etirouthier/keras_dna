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
    
    gff_file = gff_file.filter(lambda x : x.fields[2] in annotation_list)
    
    for gff in gff_file:
        chroms.append(gff.chrom)
        starts.append(gff.start)
        stops.append(gff.stop)
        labels.append(gff.fields[2])
        
        if strands:
            strands.append(gff.strand)
    
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