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
from distutils.spawn import find_executable

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def bed_to_df(bedfile, annotation_list):
    bedfile = pybedtools.BedTool(bedfile)
    
    assert len(annotation_list) == 1, \
    """A .bed file can only display the position for one type of 
    annotation."""
    
    chroms, starts, stops, labels = [], [], [], []
    strands = None

    if hasattr(bedfile[0], 'strand'):
        if bedfile[0].strand in ['+', '-']:
            strands = []

    for bed in bedfile:
        chroms.append(bed.chrom)
        starts.append(bed.start)
        stops.append(bed.stop)
        labels.append(annotation_list[0])

        if hasattr(bed, 'strand'):
            if bed.strand in ['+', '-']:
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
    if not gff_file[1].strand == '.':
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

def bedGraph_to_df(bedGraph, chrom_size, path=None):
    assert find_executable('bedGraphToBigWig')\
    or find_executable(os.path.join(path, 'bedGraphToBigWig')),\
    '''bedGraphToBigWig is not available to use wig files.
    If you are using a conda environment you can download it by running :

    conda install -c bioconda ucsc-bedgraphtobigwig.

    Otherwise run keras_dna.get_ucsc in a terminal to download it from UCSC.
    After that you will need to specify the path to the directory where you
    install the script with the keyword path_to_ucsc.'''
    assert os.path.exists(bedGraph),\
    """The bedGraph file {} does not exists""".format(bedGraph)
    if path:
        output = os.system(path + '/bedGraphToBigWig ' + bedGraph + \
              ' ' + chrom_size + ' ' + bedGraph[:-8] + 'bw')
    else:
         output = os.system('bedGraphToBigWig ' + bedGraph + \
                  ' ' + chrom_size + ' ' + bedGraph[:-8] + 'bw')
    
    if output != 0:
        print("An error has occured. Returned os error code {}".format(output))

def wig_to_df(wig, chrom_size, path=None):
    assert find_executable('wigToBigWig')\
    or find_executable(os.path.join(path, 'wigToBigWig')),\
    '''wigToBigWig is not available to use wig files.
    If you are using a conda environment you can download it by running :

    conda install -c bioconda ucsc-wigtobigwig.

    Otherwise run keras_dna.get_ucsc in a terminal to download it from UCSC.
    After that you will need to specify the path to the directory where you
    install the script with the keyword path_to_ucsc.'''
    assert os.path.exists(wig),\
    """The wig file {} does not exists""".format(wig)
    if path:
         output = os.system(path + '/wigToBigWig ' + wig + \
              ' ' + chrom_size + ' ' + wig[:-3] + 'bw')
    else:
         output = os.system('wigToBigWig ' + wig + \
                  ' ' + chrom_size + ' ' + wig[:-3] + 'bw')
    
    if output != 0:
        print("An error has occured. Returned os error code {}".format(output))

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
    
    if isinstance(labels, np.ndarray):
        labels = labels[:, ::-1]
    
    if bbi_seqs is not None:
        bbi_seqs = bbi_seqs[:, ::-1]
    
        return seqs, bbi_seqs, labels
    else:
        return seqs, labels

def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):  
    """ 
        Take a numpy array and return a view of this array after applying a rolling window.

        This takes a numpy and cut it in several pieces with the size, the stride and the 
        axes asked as needed. You may want to use it in order to create a set of sequences
        from an array.

        :param array: The array we want to cut
        :param window: The length of the window
        :param asteps: The stride between two window applied
        :param wsteps: The stride whitin the window
        :param axes: The axe on which to apply the rolling window
        :param toend: Weither or not to finish the cut
        :type array: numpy array
        :type window: int or tuple
        :type asteps: int or tuple
        :type wsteps: int or tuple
        :type axes: int
        :type toend: boolean
        :return: The view of the array
        :rtype: numpy array

        :Example:

        >>> a = numpy.array([0,1,2,3,4,5])
        >>> rolling_window(a, window = 2, asteps = 2, wsteps = None)
        array([[0,1],
               [2,3],
               [4,5]])
        >>> rolling_window(a, window = 2, asteps = None, wsteps = 2)
        array([[0,2],
               [1,3],
               [2,4]
               [3,5]])
        >>> rolling_window(a, window = 5, asteps = 2, wsteps = None)
        array([[0,1,2,3,4]])

        .. warning:: Be carreful about the combination of window, wsteps and asteps that may raise 
                     ValueError. This function forces the window to be of the asked size and thus 
                     may stop the application of the window before the end.
    """        

    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int) # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger than 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.") 

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps
        
        if np.any(asteps < 1):
             raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
             raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger than the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any \"old\" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps
    
    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _
        
        new_shape = np.zeros(len(shape)*2, dtype=int)
        new_strides = np.zeros(len(shape)*2, dtypenucleotid=int)
        
        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides
    
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)



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
        -ess and their arguments.
        """
        if self.called_args:
            args_instance = getattr(self.instance, self.called_args)
            args_dico = args_instance.command_dict.get_details()
            args_dico.update({str(type(self.instance))[8 : -2] : self.__call__()['args']})
            return args_dico
        else:
            args_dico = self.as_input()
            if hasattr(self.instance, 'length'):
                args_dico.update({'length' : self.instance.length})
            return {str(type(self.instance))[8 : -2] : args_dico}
