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
        

class Correlate(object):
    """
    info:
        doc: >
            Class usefull to evaluate a continuous model with the correlation on
            a specific cellular type and annotation if the model outputs several
            of them.
    args:
        cell_idx:
            Index of the cellular type to evaluate.
        ann_idx:
            Index of the annotation.
        nb_cell_type:
            Number of different "cellular type" on which the model was trained
            on. An evaluation will be performed on each of them independently.
        nb_annotation_type:
            Number of different annotation that the model is trained to predict
            simultaneously, an evaluation will be performed for each of them
            independently.
    """
    def __init__(self,
                 cell_idx,
                 idx,
                 nb_types,
                 nb_annotation):
        self.cell_idx = int(cell_idx)
        self.idx = int(idx)
        self.nb_types = int(nb_types)
        self.nb_annotation = int(nb_annotation)

    def metric(self,
               y_pred,
               y_true):
        y_pred = self._project(y_pred)
        y_true = self._project(y_true)

        X = y_true - K.mean(y_true)
        Y = y_pred - K.mean(y_pred)

        sigma_XY = K.sum(X*Y)
        sigma_X = K.sqrt(K.sum(X*X))
        sigma_Y = K.sqrt(K.sum(Y*Y))

        return sigma_XY/(sigma_X*sigma_Y + K.epsilon())

    def _project(self, y):
        if self.nb_types == 1 and self.nb_annotation != 1:
            if len(K.int_shape(y)) == 4:
                y = y[:, :, self.cell_idx, self.idx]
            elif len(K.int_shape(y)) == 3:
                y = y[:, :, self.idx]

        elif self.nb_types != 1 and self.nb_annotation == 1:
            if len(K.int_shape(y)) == 4:
                y = y[:, :, self.cell_idx, self.idx]
            elif len(K.int_shape(y)) == 3:
                y = y[:, :, self.cell_idx]

        elif self.nb_types == 1 and self.nb_annotation == 1:
            if len(K.int_shape(y)) == 4:
                y = y[:, :, 0, 0]
            elif len(K.int_shape(y)) == 3:
                y = y[:, :, 0]

        return y
    