#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:30:24 2020

@author: routhier
"""

import numpy as np
import tensorflow.keras as keras


def get_maximum_first_layer_activations(first_layer_model):
    first_layer, biases = first_layer_model.layers[0].get_weights()
    return np.sum(np.max(np.abs(first_layer), axis=1), axis=0) + biases

def get_maximum_second_layer_activations(first_layers_model):
    max_first_layer = get_maximum_first_layer_activations(first_layers_model)

    second_layer, biases = first_layers_model.layers[-1].get_weights()
    activation = first_layers_model.layers[0].activation
    max_first_layer = activation(max_first_layer).numpy()

    if len(first_layers_model.layers) == 4:
        gamma, beta, mean, std = first_layers_model.layers[2].get_weights()
        std[std < 10e-5] = 1
        max_first_layer = gamma * (max_first_layer - mean) / std + beta

    max_first_layer = np.tile(max_first_layer, [len(second_layer), second_layer.shape[2], 1])
    max_first_layer = np.swapaxes(max_first_layer, 1, 2)

    second_layer[second_layer < 0] = 0

    max_second_layer = max_first_layer * second_layer
    return np.sum(np.sum(max_second_layer, axis=1), axis=0) + biases

def find_pfm_on_batch(sequences,
                      first_layer_model,
                      pool_size,
                      threshold=0.5,
                      layer='first_layer'):
    if layer == 'first_layer':
        max_activation = get_maximum_first_layer_activations(first_layer_model)
    else:
        max_activation = get_maximum_second_layer_activations(first_layer_model)
    seq_shape = np.array(sequences.shape)
    
    if len(seq_shape) == 4:
        axis = np.where(seq_shape == 1)[0][0]
        if axis == 2:
            sequences = sequences[:, :, 0, :]
        else:
            sequences = sequences[:, :, :, 0]
    
    if layer == 'first_layer':
        length = len(first_layer_model.layers[0].get_weights()[0])
    else:
        if len(first_layer_model.layers) == 4:
            conv_idx = 3
        else:
            conv_idx = 2
        length = len(first_layer_model.layers[conv_idx].get_weights()[0]) * pool_size \
                + len(first_layer_model.layers[0].get_weights()[0]) - 1 

    activations = first_layer_model.predict(sequences)
    activations = np.swapaxes(activations, 0, 2)
    activations = np.swapaxes(activations, 1, 2)
    
    max_activation = np.repeat(max_activation,
                               activations.shape[1] * activations.shape[2]).reshape(activations.shape)

    kernel_indexes, seq_indexes, positions = np.where(activations >= threshold * max_activation)
    positions = positions * pool_size
    
    number_activator_seq = [len(np.unique(seq_indexes[kernel_indexes == i]))\
                            for i in range(len(activations))]
    number_activations = [len(seq_indexes[kernel_indexes == i])\
                          for i in range(len(activations))]
    return np.concatenate([extract_kernel_on_batch(kernel, sequences, kernel_indexes, seq_indexes, positions, length) for kernel in range(len(max_activation))], 0),\
np.array(number_activator_seq), np.array(number_activations)

def extract_kernel_on_batch(kernel, sequence, kernel_indexes, seq_indexes, positions, length):
    if len(seq_indexes[kernel_indexes == kernel]) > 0:
        pfm = np.mean(sequence[np.repeat(seq_indexes[kernel_indexes == kernel], length).reshape((len(seq_indexes[kernel_indexes == kernel]), length)),
                     np.concatenate([np.expand_dims(np.arange(pos, pos + length), 0) for pos in positions[kernel_indexes == kernel]], axis=0)], axis=0)
        return np.expand_dims(pfm, 0)
    else:
        return np.zeros((1, length, 4))

def find_pfm(generator,
             first_layer_model,
             pool_size,
             threshold=0.5,
             layer='first_layer'):
    number_of_batches = len(generator)
    new_generator = generator()

    for i in range(number_of_batches):
        if i == 0:
            sequences = next(new_generator)[0]
            nb_seq = len(sequences)
            pfms, nb_act_seqs, nb_acts = find_pfm_on_batch(sequences,
                                                           first_layer_model,
                                                           pool_size,
                                                           threshold,
                                                           layer)
            normalizer = np.ones(pfms.shape)
            normalizer[np.sum(np.sum(pfms, axis=2), axis=1) == 0] = 0

        else:
            sequences = next(new_generator)[0]
            nb_seq += len(sequences)
            pfm, nb_act_seq, nb_act = find_pfm_on_batch(sequences,
                                                        first_layer_model,
                                                        pool_size,
                                                        threshold,
                                                        layer)
            pfms += pfm
            nb_act_seqs += nb_act_seq
            nb_acts += nb_act
            counter = np.ones(pfms.shape)
            counter[np.sum(np.sum(pfm, axis=2), axis=1) == 0] = 0
            normalizer += counter

    return pfms / normalizer, nb_act_seqs / nb_seq, nb_acts / nb_act_seqs

def create_logos(pfm):
    pfm[pfm == 0] = 0.001
    return  pfm * (2 + np.concatenate([np.expand_dims(np.sum(pfm * np.log2(pfm), axis=-1), -1) for i in range(4)], -1))

def get_conv_parameters(model, idx):
    if isinstance(model.layers[idx], keras.layers.Conv1D):
        length_filters, alphabet, nb_filters = model.layers[idx].get_weights()[0].shape
        _, length_seq, alphabet = keras.backend.int_shape(model.input)
        return length_seq, length_filters, alphabet, nb_filters

    elif isinstance(model.layers[idx], keras.layers.Conv2D):
        _, length_seq, dummy, alphabet = keras.backend.int_shape(model.input)
        if dummy == 1:
            length_filters, _, alphabet, nb_filters = model.layers[idx].get_weights()[0].shape
        else:
            length_filters, alphabet, _, nb_filters = model.layers[idx].get_weights()[0].shape
        return length_seq, length_filters, alphabet, nb_filters, dummy
    
def find_conv_index(model):
    layers = model.layers

    conv1d_indexes = []
    conv2d_indexes = []
    maxpool_indexes = []
    batchnorm_indexes = []

    for i, layer in enumerate(layers):
        if isinstance(layer, keras.layers.Conv1D):
            conv1d_indexes.append(i)
        if isinstance(layer, keras.layers.Conv2D):
            conv2d_indexes.append(i)
        if isinstance(layer, keras.layers.BatchNormalization):
            batchnorm_indexes.append(i)

    for i, dico in enumerate(model.get_config()['layers']):
        if dico['class_name'] == 'MaxPooling1D' or\
        dico['class_name'] == 'MaxPooling2D':
            maxpool_indexes.append(i)
    return conv1d_indexes, conv2d_indexes, maxpool_indexes, batchnorm_indexes

def add_conv_layer(model,
                   first_layers_model,
                   idx,
                   dimension,
                   activation,
                   input_shape=True):
    if dimension == 'Conv1D':
        length_seq, length_filters, alphabet, nb_filters = \
        get_conv_parameters(model, idx)
    elif dimension == 'Conv2D':
        length_seq, length_filters, alphabet, nb_filters, dummy = \
        get_conv_parameters(model, idx)
        
    if input_shape:
        input_shape = (length_seq, alphabet)
        first_layers_model.add(keras.layers.Conv1D(filters=nb_filters,
                                               kernel_size=(length_filters,),
                                               input_shape=input_shape,
                                               padding='valid',
                                               activation=activation))
    else:
        first_layers_model.add(keras.layers.Conv1D(filters=nb_filters,
                                               kernel_size=(length_filters,),
                                               padding='valid',
                                               activation=activation))

    if dimension == 'Conv1D':
        first_layers_model.layers[-1].set_weights(model.layers[idx].get_weights())
    elif dimension == 'Conv2D':
        weights, biases = model.layers[idx].get_weights()
        if dummy == 1:
            first_layers_model.layers[-1].set_weights([weights[:, 0, :, :],
                                                       biases])
        else:
            first_layers_model.layers[-1].set_weights([weights[:, :, 0, :],
                                                       biases])

def create_first_layer_model(model, layer='first_layer'):
    first_layers_model = keras.models.Sequential()
    conv1d_indexes, conv2d_indexes, maxpool_indexes, batchnorm_indexes = find_conv_index(model)

    if conv1d_indexes:
        if layer == 'first_layer':
            activation='linear'
        else:
            activation = model.layers[conv1d_indexes[0]].activation
        add_conv_layer(model,
                       first_layers_model,
                       conv1d_indexes[0],
                       'Conv1D',
                       activation)
    elif conv2d_indexes:
        if layer == 'first_layer':
            activation='linear'
        else:
            activation = model.layers[conv2d_indexes[0]].activation
        add_conv_layer(model,
                       first_layers_model,
                       conv1d_indexes[0],
                       'Conv2D',
                       activation)
    else:
        raise ValueError('''The model must contain a convolutional layer''')
    
    if layer == 'first_layer':
        return (first_layers_model, 1)
    else:
        if maxpool_indexes:
            pool_size = model.get_config()['layers'][maxpool_indexes[0]]\
            ['config']['pool_size'][0]
        else:
            pool_size = 1

        first_layers_model.add(keras.layers.MaxPooling1D(pool_size, padding='valid'))

        if batchnorm_indexes:
            idx = batchnorm_indexes[0]
            first_layers_model.add(keras.layers.BatchNormalization())
            first_layers_model.layers[-1].set_weights(model.layers[idx].get_weights())
        
        if conv1d_indexes:
            add_conv_layer(model, first_layers_model, conv1d_indexes[1],
                           'Conv1D', 'linear', False)
        else:
            add_conv_layer(model, first_layers_model, conv2d_indexes[1],
                           'Conv2D', 'linear', False)
        return (first_layers_model, pool_size)

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
