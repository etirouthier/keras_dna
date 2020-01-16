# MultiGenerator

## Introduction

`MultiGenerator` is a class useful to train a model on several species at the same time or on both the positive and negative strand of the data. It uses several datasets as seen previously ([Dataset](dataset.md)) to generate batches of data that are a mix of the data coming from every dataset.

## Creating an instance

To create an instance of a `MutliGenerator` one need to create all the dataset that will be used and to ensure that the shape of the inputs and the labels are the same in all those dataset. The mandatory arguments are `batch_size` and `dataset_list`.

```python
from keras_dna import SeqIntervalDl, MultiGenerator

dataset1 = SeqIntervalDl(fasta_file='species1.fa',
                         annotation_files=['ann1.gff'],
                         annotation_list=['binding site'])

### Same keyword except fasta and annotation file
dataset2 = SeqIntervalDl(fasta_file='species2.fa',
                         annotation_files=['ann2.gff'],
                         annotation_list=['binding site'])

generator = MultiGenerator(batch_size=64,
                           dataset_list=[dataset1, dataset2]) 
```

**Warning :** incoherent choice of dataset (a stringseq and a seq for instance) is not verified before processing and will lead to an error during training.

## Changing labels shape

Use the keyword `output_shape` to readapt the labels shape to your need (same principle as in [Generator](generator.md).

## Changing the number if instance per dataset

If one need to control the number of instance that the generator will return for every dataset, one can use the keyword `inst_per_dataset`. A list of integer representing the number of instance for every dataset (same order) need to be passed. The default behaviour is to generate all the data available (which can lead to a bias toward one species).

```python
from keras_dna import SeqIntervalDl, MultiGenerator

dataset1 = SeqIntervalDl(fasta_file='species1.fa',
                         annotation_files=['ann1.gff'],
                         annotation_list=['binding site'])

### Same keyword except fasta and annotation file
dataset2 = SeqIntervalDl(fasta_file='species2.fa',
                         annotation_files=['ann2.gff'],
                         annotation_list=['binding site'])

generator = MultiGenerator(batch_size=64,
                           dataset_list=[dataset1, dataset2],
                           inst_per_dataset=[10000, 10000]) 
```

-----------------------------
