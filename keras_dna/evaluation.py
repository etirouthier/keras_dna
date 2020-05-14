#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:33:49 2019

@author: routhier
"""

import tensorflow as tf
from tensorflow.keras import backend as K


class Auc(object):
    """
    info:
        doc: >
            Class usefull to evaluate a sparse model with the AUROC metric on
            a specific cellular type and annotation if the model outputs several
            of them.
    args:
        curve:
            {'ROC', 'PR'} two available metrics.
            default='ROC'
    """
    def __init__(self,
                 curve='ROC'):
        assert curve in ['ROC', 'PR'], \
        """ The two available metrics are 'ROC' and 'PR' """
        self.curve = curve
        
    def metric(self,
               y_pred,
               y_true):
        auc, update_op = tf.metrics.auc(y_pred, y_true, curve=self.curve)
        K.get_session().run(tf.local_variables_initializer())
        
        with tf.control_dependencies([update_op]):
            auc = tf.identity(auc)
        return auc


def correlate(y_pred, y_true, cell_idx, idx, nb_types, nb_annotation):
    y_pred = _project(y_pred, cell_idx, idx, nb_types, nb_annotation)
    y_true = _project(y_true, cell_idx, idx, nb_types, nb_annotation)

    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)

    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))

    return sigma_XY/(sigma_X*sigma_Y + K.epsilon())

def _project(y, cell_idx, idx, nb_types, nb_annotation):
    if nb_types == 1 and nb_annotation != 1:
        if len(K.int_shape(y)) == 4:
            y = y[:, :, cell_idx, idx]
        elif len(K.int_shape(y)) == 3:
            y = y[:, :, idx]

    elif nb_types != 1 and nb_annotation == 1:
        if len(K.int_shape(y)) == 4:
            y = y[:, :, cell_idx, idx]
        elif len(K.int_shape(y)) == 3:
            y = y[:, :, cell_idx]

    elif nb_types == 1 and nb_annotation == 1:
        if len(K.int_shape(y)) == 4:
            y = y[:, :, 0, 0]
        elif len(K.int_shape(y)) == 3:
            y = y[:, :, 0]

    return y
