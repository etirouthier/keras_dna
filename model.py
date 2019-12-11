#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:35:42 2019

@author: routhier
"""

import json
from copy import deepcopy
import pyBigWig


from keras.utils.io_utils import H5Dict
from keras.models import load_model


from generators import Generator, MultiGenerator, PredictionGenerator
from sequence import SeqIntervalDl, StringSeqIntervalDl


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
            case of a MultiGeneratoe).
            default=None            
        validation_chr:
            The chromosome used as validation set (if no generator_val passed,
            the generator_val will be the same as generator_train except for
            the incl_chromosomes)
    """
    def __init__(self,
                 model,
                 generator_train,
                 generator_val=None,
                 validation_chr=None):
        self.model = model
        self.generator_train = generator_train

        if generator_val:
           self.generator_val = generator_val
        elif validation_chr:
            command_dict = deepcopy(self.generator_train.command_dict.as_input())
            command_dict['incl_chromosomes'] = validation_chr
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
        self._update_hdf5(h5dict, self.generator_val.command_dict, 'val')

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

                        self._export_to_big_wig(path + '_cell_number{}_{}'\
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
                                try:
                                    array = prediction[:, 0]
                                except IndexError:
                                    array = prediction
                        else:
                            array = prediction[:, cell_idx, idx]

                        self._export_to_bigwig(path + '_cell_number{}_{}.bw    '\
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
            values = array[int(row.first_index) : int(row.last_index)]\
                               .reshape(array.shape[0] * array.shape[1])
            values = values.astype(float).tolist()
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

    arguments_val = json.loads(h5dict['arguments_val'].decode('utf8'))
    generator_val = load_generator(arguments_val)

    wrapped_model = ModelWrapper(model, generator_train, generator_val) 
    h5dict.__exit__()
    return wrapped_model

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































