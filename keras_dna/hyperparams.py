#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:30:24 2020

@author: routhier
"""
import numpy as np

def create_usual_parameters(parameters,
                            name,
                            test_initializer=True,
                            test_constraint=True,
                            test_regularizer=True,
                            test_activation=True):
    if test_activation:
        parameters.update({name + '_activation' : {'values' : ['relu', 'linear', 'selu',
                                                               'elu', 'exponential']}})
    if test_initializer:
        parameters.update({name + '_kernel_initializer' : {'values' : [{'class_name': 'RandomNormal',
                                                                        'config': {'mean': 0.0,
                                                                                   'stddev': 0.05,
                                                                                   'seed': None}},
                                                                       {'class_name': 'RandomUniform',
                                                                        'config': {'minval': -0.05,
                                                                                   'maxval': 0.05,
                                                                                   'seed': None}},
                                                                       {'class_name': 'TruncatedNormal',
                                                                        'config': {'mean': 0.0,
                                                                                   'stddev': 0.05,
                                                                                   'seed': None}},
                                                                       {'class_name': 'Zeros',
                                                                        'config': {}},
                                                                       {'class_name': 'Ones',
                                                                        'config': {}},
                                                                       {'class_name': 'GlorotNormal',
                                                                        'config': {'seed': None}},
                                                                       {'class_name': 'GlorotUniform',
                                                                        'config': {'seed': None}}]}})
        parameters.update({name + '_bias_initializer' :   {'values' : [{'class_name': 'RandomNormal',
                                                                        'config': {'mean': 0.0,
                                                                                   'stddev': 0.05,
                                                                                   'seed': None}},
                                                                       {'class_name': 'RandomUniform',
                                                                        'config': {'minval': -0.05,
                                                                                   'maxval': 0.05,
                                                                                   'seed': None}},
                                                                       {'class_name': 'TruncatedNormal',
                                                                        'config': {'mean': 0.0,
                                                                                   'stddev': 0.05,
                                                                                   'seed': None}},
                                                                       {'class_name': 'Zeros',
                                                                        'config': {}},
                                                                       {'class_name': 'Ones',
                                                                        'config': {}},
                                                                       {'class_name': 'GlorotNormal',
                                                                        'config': {'seed': None}},
                                                                       {'class_name': 'GlorotUniform',
                                                                        'config': {'seed': None}}]}})
    if test_regularizer:
        parameters.update({name + '_kernel_regularizer' : {'values' : ['l1',
                                                                       'l2',
                                                                       'l1_l2',
                                                                       None]}})
        parameters.update({name + '_bias_regularizer' :  {'values' : ['l1',
                                                                       'l2',
                                                                       'l1_l2',
                                                                       None]}})
    if test_constraint:
        parameters.update({name + '_kernel_constraint' : {'values' : [{'class_name': 'MaxNorm',
                                                                       'config': {'max_value': 2,
                                                                                  'axis': 0}},
                                                                      {'class_name': 'MaxNorm',
                                                                       'config': {'max_value': 1,
                                                                                  'axis': 0}},
                                                                      {'class_name': 'MinMaxNorm',
                                                                       'config': {'min_value': 0.0,
                                                                                  'max_value': 1.0,
                                                                                  'rate': 1.0,
                                                                                  'axis': 0}},
                                                                      {'class_name': 'MinMaxNorm',
                                                                       'config': {'min_value': 0.0,
                                                                                  'max_value': 2.0,
                                                                                  'rate': 1.0,
                                                                                  'axis': 0}},
                                                                      {'class_name': 'NonNeg',
                                                                       'config': {}},
                                                                      {'class_name': 'UnitNorm',
                                                                       'config': {'axis': 0}},
                                                                      None]}})
        parameters.update({name + '_bias_constraint' : {'values' : [{'class_name': 'MaxNorm',
                                                                     'config': {'max_value': 2,
                                                                                'axis': 0}},
                                                                    {'class_name': 'MaxNorm',
                                                                     'config': {'max_value': 1,
                                                                                'axis': 0}},
                                                                    {'class_name': 'MinMaxNorm',
                                                                     'config': {'min_value': 0.0,
                                                                                'max_value': 1.0,
                                                                                'rate': 1.0,
                                                                                'axis': 0}},
                                                                    {'class_name': 'MinMaxNorm',
                                                                     'config': {'min_value': 0.0,
                                                                                'max_value': 2.0,
                                                                                'rate': 1.0,
                                                                                'axis': 0}},
                                                                    {'class_name': 'NonNeg',
                                                                     'config': {}},
                                                                    {'class_name': 'UnitNorm',
                                                                     'config': {'axis': 0}},
                                                                    None]}})
    return parameters

def create_parameters_conv1d(dico_layer,
                             test_initializer=True,
                             test_constraint=True,
                             test_regularizer=True,
                             test_activation=True,
                             test_dilation=True):
    name = dico_layer['config']['name']
    parameters = {name + '_filters' : {'values' : [2, 4, 8, 16, 32, 64, 128]},
                  name + '_kernel_size' : {'values' : [4, 8, 12, 16, 20]}}
    if test_dilation:
        parameters.update({name + '_dilation' : {'values' : [1, 2, 4, 8]}})

    parameters = create_usual_parameters(parameters,
                                         name,
                                         test_initializer,
                                         test_constraint,
                                         test_regularizer,
                                         test_activation)
    return parameters

def create_parameters_dense(dico_layer,
                            test_initializer=True,
                            test_constraint=True,
                            test_regularizer=True,
                            test_activation=True):
    name = dico_layer['config']['name']
    parameters = {name + '_units' : {'values' : [50, 100, 150, 200, 250]}}

    parameters = create_usual_parameters(parameters,
                                         name,
                                         test_initializer,
                                         test_constraint,
                                         test_regularizer,
                                         test_activation)
    return parameters

def create_parameters_conv2d(dico_layer,
                             test_initializer=True,
                             test_constraint=True,
                             test_regularizer=True,
                             test_activation=True,
                             test_dilation=True):
    name = dico_layer['config']['name']
    parameters = {name + '_filters' : {'values' : [2, 4, 8, 16, 32, 64, 128]}}

    kernel_size = dico_layer['config']['kernel_size']

    # if the conv layer is the first one the alphabet can be the second or the third
    if 'batch_input_shape' in dico_layer['config']:
        batch_input_shape = dico_layer['config']['batch_input_shape']

        if batch_input_shape[2] == 1:
            parameters.update({name + '_kernel_size' : {'values' : [[2, 1],
                                                                    [4, 1],
                                                                    [8, 1],
                                                                    [16, 1],
                                                                    [32, 1],
                                                                    [64, 1], 
                                                                    [128, 1]]}})
        elif batch_input_shape[2] == 4:
            parameters.update({name + '_kernel_size' : {'values' : [[2, 4],
                                                                    [4, 4],
                                                                    [8, 4],
                                                                    [16, 4],
                                                                    [32, 4],
                                                                    [64, 4], 
                                                                    [128, 4]]}})
    else:
        if kernel_size == [1, 1]:
            pass
            # this kernel_size is special, we do not want to optimize the size
        else:
            parameters.update({name + '_kernel_size' : {'values' : [[2, 1],
                                                                    [4, 1],
                                                                    [8, 1],
                                                                    [16, 1],
                                                                    [32, 1],
                                                                    [64, 1], 
                                                                    [128, 1]]}})

    if test_dilation:
        parameters.update({name + '_dilation' : {'values' : [[1, 1],
                                                             [2, 1],
                                                             [4, 1], 
                                                             [8, 1]]}})

    parameters = create_usual_parameters(parameters,
                                         name,
                                         test_initializer,
                                         test_constraint,
                                         test_regularizer,
                                         test_activation)
    return parameters

def create_parameters_maxpooling1d(dico_layer):
    name = dico_layer['config']['name']
    parameters = {name + '_pool_size' : {'values' : [2, 4, 6, 8]}}
    return parameters

def create_parameters_maxpooling2d(dico_layer):
    name = dico_layer['config']['name']
    parameters = {name + '_pool_size' : {'values' : [[2, 1],
                                                     [4, 1],
                                                     [6, 1],
                                                     [8, 1]]}}
    return parameters

def create_parameters_dropout(dico_layer):
    name = dico_layer['config']['name']
    parameters = {name + '_rate' : {'values' : np.arange(0, 1, 0.2)}}
    return parameters

def create_parameters_one_layer(dico_layer,
                                test_initializer=True,
                                test_constraint=True,
                                test_regularizer=True,
                                test_activation=True,
                                test_dilation=True,
                                test_maxpooling=True):
    class_name = dico_layer['class_name']

    if class_name == 'Conv1D':
        return create_parameters_conv1d(dico_layer,
                                        test_initializer,
                                        test_constraint,
                                        test_regularizer,
                                        test_activation,
                                        test_dilation)
    if class_name == 'Conv2D':
        return create_parameters_conv2d(dico_layer,
                                        test_initializer,
                                        test_constraint,
                                        test_regularizer,
                                        test_activation,
                                        test_dilation)
    if class_name == 'Dense':
        return create_parameters_dense(dico_layer,
                                       test_initializer,
                                       test_constraint,
                                       test_regularizer,
                                       test_activation)
    if class_name == 'Dropout':
        return create_parameters_dense(dico_layer)

    if test_maxpooling:
        if class_name == 'MaxPooling1D':
            return create_parameters_maxpooling1d(dico_layer)
        if class_name == 'MaxPooling2D':
            return create_parameters_maxpooling2d(dico_layer)

def update_sweep_config(config,
                        sweep_config,
                        test_initializer=True,
                        test_constraint=True,
                        test_regularizer=True,
                        test_activation=True,
                        test_dilation=True,
                        test_maxpooling=True):
    for dico_layer in config['config']['layers']:
        try:
            sweep_config['parameters'].update(create_parameters_one_layer(dico_layer,
                                                                          test_initializer,
                                                                          test_constraint,
                                                                          test_regularizer,
                                                                          test_activation,
                                                                          test_dilation))
        except TypeError:
            pass
    return sweep_config

def change_config_layer(dico_layer, wandb_config):
    layer_name = dico_layer['config']['name']

    for parameter in dico_layer['config'].keys():
        if layer_name + '_' + parameter in wandb_config:
            dico_layer['config'][parameter] = wandb_config[layer_name + '_' + parameter]
    return dico_layer

def change_config(config, wandb_config, freeze_layer=None):
    dico_layers = config['config']['layers']
    if freeze_layer:
        dico_layers = np.delete(dico_layers, freeze_layer)

    for dico_layer in dico_layers:
        change_config_layer(dico_layer, wandb_config)

def create_default(sweep_config):
    parameters = sweep_config['parameters']
    default_params = {}

    for parameter, values in parameters.items():
        default_params[parameter] = np.random.choice(values['values'])
    return default_params
