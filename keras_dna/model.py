#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:35:42 2019

@author: routhier
"""

import json
from copy import deepcopy
import pyBigWig
import numpy as np


from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.models import Model
from tensorflow.keras import Input


from .generators import Generator, MultiGenerator, PredictionGenerator
from .sequence import SeqIntervalDl, StringSeqIntervalDl
from .evaluation import Auc, Correlate
from .layers import Project1D
from .keras_utils import H5Dict
    

class ModelWrapper(object):
    """
    info:
        doc: >
            Wrap a keras model and its training generator. Anable to train,
            evaluate and predict with the model. Both trained model, and the
            generator configuration are saved in the same hdf5 file and can be
            load.
    args:
        model:
            A compiled keras model to train.
        generator_train:
            A Generator or MultiGenerator instance to train the model.
        generator_val:
            The generator used to generate the validation set (usefull in the
            case of a MultiGenerator).
            default=None            
        validation_chr:
            The chromosome used as validation set (if no generator_val passed,
            the generator_val will be the same as generator_train except for
            the incl_chromosomes)
        weights_val:
            boolean, weither or not to include the training weights in
            validation.
            default=False
    """
    def __init__(self,
                 model,
                 generator_train,
                 generator_val=None,
                 validation_chr=None,
                 weights_val=False):
        self.model = model
        self.generator_train = generator_train

        if generator_val:
           self.generator_val = generator_val
        elif validation_chr:
            command_dict = deepcopy(self.generator_train.command_dict.as_input())
            command_dict['incl_chromosomes'] = validation_chr

            if not weights_val:
                command_dict['weighting_mode'] = None
            self.generator_val = Generator(**command_dict)

    def train(self,
              epochs,
              steps_per_epoch=None,
              validation_steps=None,
              *args,
              **kwargs):
        if not steps_per_epoch:
            steps_per_epoch = len(self.generator_train)
        if not validation_steps:
            validation_steps = len(self.generator_val)

        self.model.fit_generator(generator = self.generator_train(),
                                 steps_per_epoch = steps_per_epoch, 
                                 epochs = epochs,
                                 validation_data = self.generator_val(), 
                                 validation_steps = validation_steps, 
                                 *args,
                                 **kwargs)

    def _update_hdf5(self, h5dict, arguments, dataset):
        if isinstance(arguments, list):
            if 'sequence.SeqIntervalDl' in arguments[0].get_details():
                dico = {'type': 'Multi Seq',
                        'arguments':  [com_dict.as_input() for com_dict in arguments]}
                dico['arguments'][-1].pop('dataset_list', None)
                h5dict['arguments_' + dataset] = json.dumps(dico).encode('utf8')
            else:
                dico = {'type': 'Multi StringSeq',
                        'arguments':  [com_dict.as_input() for com_dict in arguments]}
                dico['arguments'][-1].pop('dataset_list', None)
                h5dict['arguments_' + dataset] = json.dumps(dico).encode('utf8')
        else:
            h5dict['arguments_' + dataset] = json.dumps({'type': 'Generator',
            'arguments': arguments.as_input()}).encode('utf8')

    def save(self,
             path,
             save_model=False):
        """
        Function usefull to store both the keras model and a dictionary usefull
        to reconstruct the generator. Usually the model will already be stored
        during the training and the path need to be the storing path.
        """
        if save_model:
            self.model.save(path)

        h5dict = H5Dict(path)
        self._update_hdf5(h5dict, self.generator_train.command_dict, 'train')
        
        try:
            self._update_hdf5(h5dict, self.generator_val.command_dict, 'val')
        except AttributeError:
            pass
        
    def evaluate(self,
                 incl_chromosomes=None,
                 generator_eval=None,
                 *args,
                 **kwargs):
        
        if self.generator_train.__class__.__name__ == 'MultiGenerator':
            assert generator_eval,\
            """generator_eval is needed to evaluate a MultiGenerator"""
        else:
            assert incl_chromosomes,\
            '''incl_chromosomes is needed'''
            command_dict = deepcopy(self.generator_train.command_dict.as_input())
            command_dict['incl_chromosomes'] = incl_chromosomes
            generator_eval = Generator(**command_dict)

        evaluations = self.model.evaluate_generator(generator=generator_eval(),
                                                    steps=len(generator_eval),
                                                    *args,
                                                    **kwargs)
        return {metric : evaluation for metric, evaluation in\
                zip(self.model.metric_names, evaluations)}

    def get_auc(self,
                incl_chromosomes,
                data_augmentation=True,
                fasta_file=None,
                annotation_files=None,
                curve='ROC',
                *args,
                **kwargs):
        """
        Returns the auroc score for a sparse model for every annotation on
        every cellular type.
        If data_augmentation the positive sequence are all the sequence that
        contains a whole annotation instance, if not one positive sequence per
        annotation instance.
        """
        if self.generator_train.__class__.__name__ == 'MultiGenerator':
            assert fasta_file and annotation_files,\
            """ To evaluate a MultiGenerator model, the fasta file and the
            annotation files need to be passed as inputs."""
            command_dict = self.generator_train.command_dict[0]

            if 'sequence.SeqIntervalDl' in command_dict.get_details():
                one_hot_encoding = True
            else:
                one_hot_encoding = False
            
            batch_size = self.generator_train.command_dict[-1].as_inputs()['batch_size']
        else:
            command_dict = self.generator_train.command_dict
            one_hot_encoding = command_dict.as_input()['one_hot_encoding']
            batch_size = command_dict.as_input()['batch_size']
    
        assert 'sequence.SparseDataset' in command_dict.get_details(),\
        """Auroc score is only available for sparse dataset"""

        dico = command_dict.get_details()['sequence.SparseDataset']

        assert not dico['seq2seq'],\
        """Auroc score is not available for seq2seq model"""

        if isinstance(dico['annotation_files'], list):
            nb_types = len(dico['annotation_files'])
        else:
            nb_types = 1

        if not annotation_files:
            cell_indexes = range(nb_types)
        else:
            assert len(annotation_files) == nb_types,\
            """annotation_files must be a list with the name number of entries
            as annotation_files in the generator, complete with zeros if needed
            """
            cell_indexes = np.where(np.array(annotation_files) != '0')[0]

        nb_annotation = len(dico['annotation_list'])
        
        eval_list = list()

        for cell_idx in cell_indexes:
            for idx, ann in enumerate(dico['annotation_list']):
                eval_dict = deepcopy(command_dict.as_input())

                if fasta_file:
                    eval_dict['fasta_file'] = fasta_file

                eval_dict['data_augmentation'] = data_augmentation
                if annotation_files:
                    eval_dict['annotation_files'] = annotation_files[cell_idx]
                else:
                    annotation_file =\
                    command_dict.as_input()['annotation_files']

                    if not isinstance(annotation_file, list):
                        annotation_file = [annotation_file]
                    eval_dict['annotation_files'] = annotation_file[cell_idx]

                eval_dict['incl_chromosomes'] = incl_chromosomes
                eval_dict['annotation_list'] = [ann]
                eval_dict['negative_ratio'] = 'all'
                eval_dict['batch_size'] = batch_size
                eval_dict['one_hot_encoding'] = one_hot_encoding
                eval_dict['output_shape'] = (batch_size, 1)
                metric = Auc(curve).metric

                generator_eval = Generator(**eval_dict)
                
                input_shape = next(generator_eval())[0].shape
                inputs = Input(input_shape)
                x = self.model(inputs)
                outputs = Project1D(cell_idx, idx,
                                    nb_types, nb_annotation)(x)
                model = Model([inputs, outputs])
                
                model.compile(optimizer=self.model.optimizer,
                              loss=self.model.loss,
                              metrics=[metric])

                eval_list.append({'cell_idx' : cell_idx,
                                  'annotation' : ann,
                                  'AU' + curve :\
                                  model.evaluate_generator(generator=generator_eval(),
                                                           steps=len(generator_eval),
                                                           *args,
                                                           **kwargs)[1]})
        return eval_list

    def get_correlation(self,
                        incl_chromosomes,
                        fasta_file=None,
                        annotation_files=None,
                        *args,
                        **kwargs):
        """
        Returns the correlation between the experimental and predicted coverage
        for a continuous model for every annotation on every cellular type.
        If fasta_file and annotation_file are parsed the evaluation is made
        using those data.
        """
        if self.generator_train.__class__.__name__ == 'MultiGenerator':
            assert fasta_file and annotation_files,\
            """ To evaluate a MultiGenerator model, the fasta file and the
            annotation file need to be passed as inputs."""
            command_dict = self.generator_train.command_dict[0]

            if 'sequence.SeqIntervalDl' in command_dict.get_details():
                one_hot_encoding = True
            else:
                one_hot_encoding = False

            batch_size = self.generator_train.command_dict[-1].as_inputs()['batch_size']
            output_shape = self.generator_train.command_dict[-1].as_inputs()['output_shape']
        else:
            command_dict = self.generator_train.command_dict
            one_hot_encoding = command_dict.as_input()['one_hot_encoding']
            batch_size = command_dict.as_input()['batch_size']
            output_shape = command_dict.as_input()['output_shape']

        assert 'sequence.ContinuousDataset' in command_dict.get_details(),\
        """Correlation score is only available for continuous dataset"""
        
        dico = command_dict.get_details()['sequence.ContinuousDataset']
        if dico['nb_annotation_type']:
            nb_annotation = dico['nb_annotation_type']
        else:
            nb_annotation = 1
 
        if annotation_files:
            if isinstance(dico['annotation_files'], list):
                assert len(annotation_files) == len(dico['annotation_files']),\
                """annotation_files must be a list with the name number of
                entries as annotation_files in the generator, complete with
                zeros if needed"""
            else:
                assert len(annotation_files) == 1,\
                """annotation_files must be a list with the name number of
                entries as annotation_files in the generator, complete with
                zeros if needed"""
            indexes = np.where(np.array(annotation_files) != '0')[0]
            
        else:
            if isinstance(dico['annotation_files'], list):
                indexes = range(len(dico['annotation_files']))
            else:
                indexes = [0]
        
        if isinstance(dico['annotation_files'], list):
            nb_types = len(dico['annotation_files']) // nb_annotation
        else:
            nb_types = 1
        

        eval_dict = deepcopy(command_dict.as_input())

        if fasta_file:
            eval_dict['fasta_file'] = fasta_file

        if annotation_files:
            annotation_files = np.array(annotation_files)
            annotation_files[annotation_files == '0'] = annotation_files[indexes[0]]
            eval_dict['annotation_files'] = list(annotation_files)

        eval_dict['incl_chromosomes'] = incl_chromosomes
        eval_dict['batch_size'] = batch_size
        eval_dict['one_hot_encoding'] = one_hot_encoding
        eval_dict['output_shape'] = output_shape
        eval_dict['overlapping'] = False

        generator_eval = Generator(**eval_dict)
        
        metrics = [Correlate(idx / nb_annotation,
                             idx % nb_annotation,
                             nb_types,
                             nb_annotation).metric for idx in indexes]
                          
        model = clone_model(self.model)
        model.compile(optimizer=self.model.optimizer,
                      loss=self.model.loss,
                      metrics=metrics)

        evaluations = model.evaluate_generator(generator=generator_eval(),
                                              steps=len(generator_eval),
                                              *args,
                                              **kwargs)
        
        return {'correlate_{}_{}'.format(idx / nb_annotation,\
                idx % nb_annotation) : evaluations[idx] for idx in indexes}
                
    def predict(self,
                incl_chromosomes,
                chrom_size,
                batch_size=32,
                fasta_file=None,
                export_to_path=None,
                *args,
                **kwargs):
        """
        Function designed to predict the output of the model all along the chromosome
        passed as arguments. Optionally another fasta file can be passed so that to
        predict on another species for example.

        args:
            incl_chromosomes:
                list of chromosomes to make the prediction on
            chrom_size:
                File with the size of chromosomes in two columns (tab separated)
            batch_size:
                batch size
            fasta_file:
                Name of a fasta file to predict on (if None it will be the
                file of the generator_train, or the first dataset if MultiGen).
            export_to_path:
                Path where the prediction will be exported in bigWig except if
                it is None.
                default: None
        """
        if self.generator_train.__class__.__name__ == 'MultiGenerator':
            command_dict = self.generator_train.command_dict[0]
        else:
            command_dict = self.generator_train.command_dict

        if not isinstance(incl_chromosomes, list):
            incl_chromosomes = [incl_chromosomes]

        self.pred_generator = PredictionGenerator(batch_size,
                                                  command_dict,
                                                  chrom_size,
                                                  incl_chromosomes,
                                                  fasta_file)
        prediction = self.model.predict_generator(generator=self.pred_generator(),
                                                  steps=len(self.pred_generator),
                                                  *args,
                                                  **kwargs)

        if export_to_path:
            self._multi_export_to_bigwig(export_to_path,
                                         prediction)

        return prediction

    def _multi_export_to_bigwig(self,
                                path,
                                prediction):
        if self.generator_train.__class__.__name__ == 'MultiGenerator':
            command_dict = self.generator_train.command_dict[0].get_details()
        else:
            command_dict = self.generator_train.command_dict.get_details()

        if 'sequence.SparseDataset' in command_dict:
            dico = command_dict['sequence.SparseDataset']
            
            if isinstance(dico['annotation_files'], list):
                nb_types = len(dico['annotation_files'])
            else:
                nb_types = 1

            nb_annotation = len(dico['annotation_list'])

            if dico['seq2seq']:
                for cell_idx in range(nb_types):
                    for idx, ann in enumerate(dico['annotation_list']):
                        if nb_types == 1 and nb_annotation != 1:
                            try:
                                array = prediction[:, :, cell_idx, idx]
                            except IndexError:
                                array = prediction[:, :, idx]
                        
                        elif nb_types != 1 and nb_annotation == 1:
                            try:
                                array = prediction[:, :, cell_idx, idx]
                            except IndexError:
                                array = prediction[:, :, cell_idx]
                        elif nb_types == 1 and nb_annotation == 1:
                            try:
                                array = prediction[:, :, 0, 0]
                            except IndexError:
                                try:
                                    array = prediction[:, :, 0]
                                except IndexError:
                                    array = prediction
                        else:
                            array = prediction[:, :, cell_idx, idx]

                        self._export_to_big_wig(path + '_cell_number{}_{}.bw'\
                                                .format(str(cell_idx), ann),
                                                array,
                                                1)
            else:
                for cell_idx in range(nb_types):
                    for idx, ann in enumerate(dico['annotation_list']):
                        if nb_types == 1 and nb_annotation != 1:
                            try:
                                array = prediction[:, cell_idx, idx]
                            except IndexError:
                                array = prediction[:, idx]
                        
                        elif nb_types != 1 and nb_annotation == 1:
                            try:
                                array = prediction[:, cell_idx, idx]
                            except IndexError:
                                array = prediction[:, cell_idx]
                        elif nb_types == 1 and nb_annotation == 1:
                            try:
                                array = prediction[:, 0, 0]
                            except IndexError:
                                array = prediction
                        else:
                            array = prediction[:, cell_idx, idx]

                        self._export_to_bigwig(path + '_cell_number{}_{}.bw'\
                                                .format(str(cell_idx), ann),
                                                array,
                                                1)
        elif 'sequence.ContinuousDataset' in command_dict:
            dico = command_dict['sequence.ContinuousDataset']
            if dico['nb_annotation_type']:
                nb_annotation = dico['nb_annotation_type']
            else:
                nb_annotation = 1
            
            nb_types = len(dico['annotation_files']) // nb_annotation
            
            resolution = 1
            if dico['downsampling']:
                resolution = dico['window'] // dico['tg_window']

            for cell_idx in range(nb_types):
                for idx in range(nb_annotation):
                    if nb_types == 1 and nb_annotation != 1:
                        try:
                            array = prediction[:, :, cell_idx, idx]
                        except IndexError:
                            try:
                                array = prediction[:, :, idx]
                            except IndexError:
                                pass

                    elif nb_types != 1 and nb_annotation == 1:
                        try:
                            array = prediction[:, :, cell_idx, idx]
                        except IndexError:
                            try:
                                array = prediction[:, :, cell_idx]
                            except IndexError:
                                pass

                    elif nb_types == 1 and nb_annotation == 1:
                        try:
                            array = prediction[:, :, 0, 0]
                        except IndexError:
                            try:
                                array = prediction[:, :, 0]
                            except IndexError:
                                array = prediction
                    else:
                        array = prediction[:, :, cell_idx, idx]

                    self._export_to_bigwig(path + '_cell_number{}_annotation_number{}.bw'\
                                            .format(str(cell_idx), str(idx)),
                                            array,
                                            resolution)

    def _export_to_bigwig(self,
                          path,
                          array,
                          resolution):
        try:
            chrom_size = self.pred_generator.dataset.seq_dl.dataset.chrom_size
        except AttributeError:
            chrom_size = self.pred_generator.dataset.dataset.chrom_size

        bw_header = [(str(chrom), size) for chrom, size in chrom_size.items()]

        bw_file = pyBigWig.open(path, 'w')
        bw_file.addHeader(bw_header)

        idxs = self.pred_generator.index_df
        for chrom, _ in bw_header:
            row = idxs[idxs.chrom == chrom]
            values = array[int(row.first_index) : int(row.last_index)]

            if len(array.shape) == 2:
                values = values.reshape(array.shape[0] * array.shape[1])

            values = values.astype(float)
            bw_file.addEntries(row.chrom.values[0],
                               int(row.start),
                               values=values,
                               span=int(resolution),
                               step=int(resolution))  
        bw_file.close()


def load_wrapper(path,
                 *args,
                 **kwargs):
    """
    Function used to load a stored ModelWrapper example, it will recreate both
    the generators and the keras model.
    """
    model = load_model(path, *args, **kwargs)
    h5dict = H5Dict(path)

    arguments_train = json.loads(h5dict['arguments_train'].decode('utf8'))
    generator_train = load_generator(arguments_train)
    
    try:
        arguments_val = json.loads(h5dict['arguments_val'].decode('utf8'))
        generator_val = load_generator(arguments_val)
    except AttributeError:
        pass
    try:
        wrapped_model = ModelWrapper(model, generator_train, generator_val)
    except UnboundLocalError:
        wrapped_model = ModelWrapper(model, generator_train)

    h5dict.__exit__()
    return wrapped_model

def load_generator_command(path):
    """
    Function used to load the generators input dict of a stored ModelWrapper
    """
    h5dict = H5Dict(path)

    arguments_train = json.loads(h5dict['arguments_train'].decode('utf8'))
    
    try:
        arguments_val = json.loads(h5dict['arguments_val'].decode('utf8'))
        h5dict.__exit__()
        return arguments_train, arguments_val

    except AttributeError:
        h5dict.__exit__()
        return arguments_train

def load_generator(arguments):
    if arguments['type'] == 'Multi Seq':
        dataset_list = [SeqIntervalDl(**com_dict) for com_dict in arguments['arguments'][:-1]]
        dico = arguments['arguments'][-1]
        dico.update({'dataset_list' : dataset_list})
        generator = MultiGenerator(**dico)

    elif arguments['type'] == 'Multi StringSeq':
        dataset_list = [StringSeqIntervalDl(**com_dict) for com_dict in arguments['arguments'][:-1]]
        dico = arguments['arguments'][-1]
        dico.update({'dataset_list' : dataset_list})
        generator = MultiGenerator(**dico)

    elif arguments['type'] == 'Generator':
        generator = Generator(**arguments['arguments'])

    return generator
