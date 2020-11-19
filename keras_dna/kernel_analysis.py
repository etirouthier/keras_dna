#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:30:24 2020

@author: routhier
"""

import numpy as np
import tensorflow.keras as keras


def get_maximum_first_layer_activations(first_layer_model):
    first_layer = first_layer_model.layers[0].get_weights()[0]
    return np.sum(np.max(np.abs(first_layer), axis=1), axis=0)

def find_pfm_on_batch(sequences, first_layer_model, threshold=0.5):
    max_activation = get_maximum_first_layer_activations(first_layer_model)
    seq_shape = np.array(sequences.shape)
    
    if len(seq_shape) == 4:
        axis = np.where(seq_shape == 1)[0][0]
        if axis == 2:
            sequences = sequences[:, :, 0, :]
        else:
            sequences = sequences[:, :, :, 0]

    length = len(first_layer_model.layers[0].get_weights()[0])
    seq_shape[1] = seq_shape[1] - length + 1

    max_activation = np.repeat(max_activation, seq_shape[0] * seq_shape[1]).reshape((len(max_activation), seq_shape[0], seq_shape[1]))

    activations = first_layer_model.predict(sequences)
    activations = np.swapaxes(activations, 0, 2)
    activations = np.swapaxes(activations, 1, 2)

    kernel_indexes, seq_indexes, positions = np.where(activations >= threshold * max_activation)
    return np.concatenate([extract_kernel_on_batch(kernel, sequences, kernel_indexes, seq_indexes, positions, length) for kernel in range(len(max_activation))], 0)

def extract_kernel_on_batch(kernel, sequence, kernel_indexes, seq_indexes, positions, length):
    if len(seq_indexes[kernel_indexes == kernel]) > 0:
        pfm = np.mean(sequence[np.repeat(seq_indexes[kernel_indexes == kernel], length).reshape((len(seq_indexes[kernel_indexes == kernel]), length)),
                     np.concatenate([np.expand_dims(np.arange(pos, pos + length), 0) for pos in positions[kernel_indexes == kernel]], axis=0)], axis=0)
        return np.expand_dims(pfm, 0)
    else:
        return np.zeros((1, length, 4))

def find_pfm(generator, first_layer_model, threshold=0.5):
    number_of_batches = len(generator)
    new_generator = generator()

    for i in range(number_of_batches):
        if i == 0:
            sequences = next(new_generator)[0]
            pfms = find_pfm_on_batch(sequences, first_layer_model, threshold=0.5)
            normalizer = np.ones(pfms.shape)
            normalizer[np.sum(np.sum(pfms, axis=2), axis=1) == 0] = 0

        else:
            sequences = next(new_generator)[0]
            pfm = find_pfm_on_batch(sequences, first_layer_model, threshold=0.5)
            pfms += pfm
            counter = np.ones(pfms.shape)
            counter[np.sum(np.sum(pfm, axis=2), axis=1) == 0] = 0
            normalizer += counter
    return pfms / normalizer

def create_logos(pfm):
    pfm[pfm == 0] = 0.001
    return  pfm * (2 + np.concatenate([np.expand_dims(np.sum(pfm * np.log2(pfm), axis=-1), -1) for i in range(4)], -1))

def create_first_layer_model(model):
    if isinstance(model.layers[0], keras.layers.Conv1D):
        length_filters, alphabet, nb_filters = model.layers[0].get_weights()[0].shape
        _, length_seq, alphabet = keras.backend.int_shape(model.input)

    elif isinstance(model.layers[0], keras.layers.Conv2D):
        _, length_seq, dummy, alphabet = keras.backend.int_shape(model.input)
        if dummy == 1:
            length_filters, _, alphabet, nb_filters = model.layers[0].get_weights()[0].shape
        else:
            length_filters, alphabet, _, nb_filters = model.layers[0].get_weights()[0].shape

    first_layer_model = keras.models.Sequential()
    first_layer_model.add(keras.layers.Conv1D(filters=nb_filters,
                                              kernel_size=(length_filters,),
                                              input_shape=(length_seq, alphabet),
                                              activation='linear',
                                              padding='valid'))

    if isinstance(model.layers[0], keras.layers.Conv1D):
        first_layer_model.layers[0].set_weights(model.layers[0].get_weights())

    elif isinstance(model.layers[0], keras.layers.Conv2D):
        weights, biases = model.layers[0].get_weights()
        if dummy == 1:
            first_layer_model.layers[0].set_weights([weights[:, 0, :, :], biases])
        else:
            first_layer_model.layers[0].set_weights([weights[:, :, 0, :], biases])
    return first_layer_model

def export_to_meme(logos, name, alphabet):
    with open(name, 'w') as f:
        f.write('MEME version 4\n')
        f.write('ALPHABET= ' + alphabet + '\n')
        f.write('strand= +\n')

        for i in range(len(logos)):
            f.write('\n')
            f.write('MOTIF FILTER_' + str(i) + '\n')
            f.write('letter-probability matrix: alength= 4 w= ' + str(len(logos[i])) + ' nsites= ' + str(len(logos[i])) + ' E= 1.0e-001\n')

            for line in logos[i]:
                for num in line:
                    if num == 0:
                        f.write('0.0\t')
                    else:
                        f.write(str(num)+'\t')

                f.write('\n')
