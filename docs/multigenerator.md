# MultiGenerator

## Introduction

`MultiGenerator` is a class useful to train a model on several species at the same time or on both the positive and negative strand of the genome. It uses several datasets ([Dataset](dataset.md)) to generate batches of data that mix data coming from every dataset.

## Creating an instance

To create an instance of `MutliGenerator` one needs firstly to create the datasets involved and to ensure that the shape of the inputs and the labels are the same in all those datasets. The mandatory arguments are `batch_size` and `dataset_list`.

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

Use the keyword `output_shape` to readapt the labels shape to your need (same principle as in [Generator](generators.md)).

## Changing the number of instance per dataset

To control the number of instance per dataset that the generator yields, use the keyword `inst_per_dataset`. The number of instance per dataset is passed through a list in the same order as the datasets in `dataset_list`. The default behaviour is to generate all the data available (which can leads to a bias toward one species).

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
