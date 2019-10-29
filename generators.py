#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:21:30 2019

@author: routhier
"""

import numpy as np
from kipoiseq.utils import DNA

from sequence import SeqIntervalDl

class Generator(object):
    """
    info:
        doc: >
            Generator for keras model able to yield inputs and targets in batch
            by reading into a fasta file and a annotation file. Inputs are one-
            hot-encoded.
     args:
        annotation_file:
            doc: bed3+<columns> file path containing intervals + (optionally)
            labels,
            hdf5 file with a continuous annotation along the genome, one
            dataset is one chromosome
            gff3 file with the annotations. (to be done)
        fasta_file:
            doc: Reference genome FASTA file path.
        batch_size: the size of the batch.
        num_chr_fasta:
            doc: True, the the dataloader will make sure that the chromosomes
            don't start with chr.
        label_dtype:
            doc: 'None, datatype of the task labels taken from the annotation_file.
            Example: str, int, float, np.float32'
        auto_resize_len:
            doc: None, required sequence length.
        # use_strand:
        #     doc: reverse-complement fasta sequence if bed file defines
        #     negative strand
        alphabet_axis:
            doc: axis along which the alphabet runs (e.g. A,C,G,T for DNA)
        dummy_axis:
            doc: defines in which dimension a dummy axis should be added. None
            if no dummy axis is required.
        alphabet:
            doc: >
                alphabet to use for the one-hot encoding. This defines the
                order of the one-hot encoding. Can either be a list or a
                string: 'ACGT' or ['A, 'C', 'G', 'T']. Default: 'ATGC'
        dtype:
            doc: 'defines the numpy dtype of the returned array.
            Example: int, np.int32, np.float32, float'
        ignore_targets:
            doc: if True, don't return any target variables
        args: arguments specific to the different dataloader that can be used.
        kwargs: dictionnary with specific arguments to the dataloader.
    """
    
    def __init__(self, annotation_file,
                       fasta_file,
                       batch_size=512,
                       num_chr_fasta=False,
                       label_dtype=None,
                       auto_resize_len=None,
                       # max_seq_len=None,
                       # use_strand=False,
                       alphabet_axis=1,
                       dummy_axis=None,
                       alphabet=DNA,
                       ignore_targets=False,
                       dtype=None,
                       *args,
                       **kwargs):
        self.dataset = SeqIntervalDl(annotation_file, fasta_file,
                                     num_chr_fasta=num_chr_fasta,
                                     label_dtype=label_dtype,
                                     auto_resize_len=auto_resize_len,
                                     # use_strand=use_strand,
                                     ignore_targets=ignore_targets,
                                     alphabet_axis=alphabet_axis,
                                     dummy_axis=dummy_axis,
                                     alphabet=alphabet,
                                     dtype=dtype,
                                     *args,
                                     **kwargs)
        self.batch_size = batch_size
    #@profile
    def create(self):
        """Returns a generator to train a keras model (yielding inputs and
        outputs)."""
        def generator_function(dataset, batch_size):
            indexes = np.arange(0, len(dataset), 1)
            number_of_batches = len(dataset) // batch_size
            
            while True:
            # reshuffled the train set after an epoch
                np.random.shuffle(indexes)
            
                for num in range(number_of_batches) :
                    batch_indexes = indexes[num*batch_size : (num + 1) * batch_size]
                    data = dataset[batch_indexes]
                    yield data['inputs'], data['targets']
            
        return generator_function(self.dataset, self.batch_size)
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
