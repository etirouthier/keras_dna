# Weights

## Introduction

To handle unbalanced dataset it is useful to add weights to training examples. The module `Generator` owns a way to do so (but not the module `MultiGenerator`).

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      weighting_mode='balanced')
```

## Regression

For regression problem there are three different ways to set the weights:

Automatically find bins to calculate the histogram and set the weights to have a balanced dataset:
```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      weighting_mode='balanced')
```

Passing the number of bins desired to calculate the histogram of the distribution and set weights to balance the dataset:
```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann1.bw', 'ann2.bw'],
                      weighting_mode='balanced',
                      bins=100)
```

Passing both bins and weights that one wants to apply for the training. It should be a tuples `([bins_ann1, bins_ann2, ..., bins_annN], [weights_ann1, weights_ann2, ..., weights_annN])` with weights_ann1 being a list of weights to apply to the values from ann1.bw and bins_ann1 the corresponding bins (`len(weights_ann1) = len(bins_ann1) - 1`)
```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann1.bw', 'ann2.bw'],
                      normalization_mode='min_max',
                      weighting_mode=[([[0, 0.1, 0.2, 0.3, 0.4], [0, 0.2, 0.5, 0.8, 1]],
                                       [[0.5, 2, 3, 4], [0.1, 1, 2, 4]])
```

**Warning :** to use weights with a regression generator one needs to compile the model setting `sample_weight_mode` to 'temporal'.

## Classification

For classification ona consider positive class to be all traning examples labelled positively for at least one function, and the negative class all examples negatively labelled for all functions. The distribution is the distribution of generated examples, not the distribution in the genome. One can automatically set weights to balance the dataset or pass the weights manually in a tuple `(weights_positive, weights_negative)`.

```python

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann1.gff', 'ann2.gff'],
                      annotation_list=['gene'],
                      predict='start',
                      weighting_mode='balanced')
                      
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann1.gff', 'ann2.gff'],
                      annotation_list=['gene'],
                      predict='start',
                      negative_ratio='all',
                      weighting_mode=(1000, 1))
```

----------------------------------------------
