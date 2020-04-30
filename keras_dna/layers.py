#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from tensorflow.keras.layers import Reshape, Layer


class Project1D(Layer):
    """
    Layer designed to change the output of a model with several annotation or
    cell_type to a single output.
    """
    def __init__(self,
                 cell_idx,
                 ann_idx,
                 nb_cell_type,
                 nb_annotation_type,
                 **kwargs):
        super(Project1D, self).__init__(**kwargs)
        self.cell_idx = cell_idx
        self.ann_idx = ann_idx
        self.nb_cell_type = nb_cell_type
        self.nb_annotation_type = nb_annotation_type

    def call(self, inputs):
        tensor = self._project(inputs)
        return Reshape((1,))(tensor)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def _project(self, tensor):
        if self.nb_cell_type == 1 and self.nb_annotation_type != 1:
            try:
                tensor = tensor[:, self.cell_idx, self.ann_idx]
            except ValueError:
                tensor = tensor[:, self.ann_idx]
        
        elif self.nb_cell_type != 1 and self.nb_annotation_type == 1:
            try:
                tensor = tensor[:, self.cell_idx, self.ann_idx]
            except ValueError:
                tensor = tensor[:, self.cell_idx]
        elif self.nb_cell_type == 1 and self.nb_annotation_type == 1:
            try:
                tensor = tensor[:, 0, 0]
            except ValueError:
                tensor = tensor[:, 0]
    
        else:
            tensor = tensor[:, self.cell_idx, self.ann_idx]
        return tensor
    
    
class Project2D(Layer):
    """
    Layer designed to change the output of a model with several annotation or
    cell_type to a single output for continuous annotation.
    """
    def __init__(self,
                 cell_idx,
                 ann_idx,
                 nb_cell_type,
                 nb_annotation_type,
                 **kwargs):
        super(Project2D, self).__init__(**kwargs)
        self.cell_idx = cell_idx
        self.ann_idx = ann_idx
        self.nb_cell_type = nb_cell_type
        self.nb_annotation_type = nb_annotation_type

    def call(self, inputs):
        return self._project(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[:2]

    def _project(self, tensor):
        if self.nb_types == 1 and self.nb_annotation != 1:
            try:
                tensor = tensor[:, :, self.cell_idx, self.idx]
            except ValueError:
                tensor = tensor[:, :, self.idx]

        elif self.nb_types != 1 and self.nb_annotation == 1:
            try:
                tensor = tensor[:, :, self.cell_idx, self.idx]
            except ValueError:
                tensor = tensor[:, :, self.cell_idx]
        elif self.nb_types == 1 and self.nb_annotation == 1:
            try:
                tensor = tensor[:, :, 0, 0]
            except ValueError:
                try:
                    tensor = tensor[:, :, 0]
                except ValueError:
                    tensor = tensor
        return tensor
    