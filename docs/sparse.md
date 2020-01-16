# Generating sparse data


## Introduction

`Generator` can be used to feed a keras model with the subsequence of DNA underlying all the exemple of an annotation in the genome. To do so the position of annotations need to be passed in a bed or gff file, with the name of the target annotation in a list. `Generator` will detect suffixes .bed, .gff, .gtf, .gff3 and select the positions of the desired annotation.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'])
```

The standard behaviour will be to generate all the DNA sequences underlying the desired annotation one-hot-encoded and with a length corresponding to the maximal length encountered (one window per annotation instance). It will also generate the same amounts of DNA sequences away from annotation regions. 

The one-hot-encoded DNA batch will have the shape `(batch_size, max length, 4)`. The labels will be 0 or 1 with the shape `(batch_size, nb anotation files, len(annotation_list))`, i.e here `(64, 1, 1)`.


## Model for several cell types and annotations

`Generator` can be used to feed a keras model aimed at predicting several different annotations in different cell types. In this case the input files are the fasta file corresponding to the DNA sequence of the species and a gff file for every cell type. The gff format is needed to provide the position of several annotation in one file, bed format is not adapted to do so.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['cell_type_1.gff', 'cell_type_2.gff'],
                      annotation_list=['ann1', 'ann2'])
```

The one-hot-encoded DNA batch will have the shape `(batch_size, max length, 4)`. The labels will be 0 or 1 with the shape `(batch_size, nb anotation files, len(annotation_list))`, i.e here `(64, 2, 2)`.


## Changing input length

To length of the inputs can be change with the keyword `seq_len`.

```python
from keras_dna import Generator

### Standard behaviour, take the maximal length of annotation
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'],
                      seq_len='MAXLEN')
                      
### We can ask a particular length, if longer than 'MAXLEN'
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'],
                      seq_len=299)

### We can take sequence of the exact length for every instance and complement with N to reach 'MAXLEN'
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'],
                      seq_len='real')
```

## Predicting the begining or end

It is sometimes needed to predict the position of the beginning or of the end of an annotation. For exemple to predict the TSS, starts of exons, or their ends. To do so `Generator` owns the keyword `predict`.

```python
from keras_dna import Generator

### Default behaviour
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      predict='start')

### Predicting the begining (TSS)
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      predict='start')
                      
### Predicting the end
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      predict='stop')
```
 
## Data Augmentation
 
Usually, there is far more areas on the genome that are not functionnal (or with an unknown fonctionnality). This fact leads to unbalanced dataset with fewer positive exemple if we keep the natural distribution. To assess this fact one can multiply the positive exemple with the data augmentation procedure, that is to say that all the window that contains the annotation are labeled as positive. This functionnality is available with the keyword `data_augmentation`:
 
```python
from keras_dna import Generator

### All the window containing an annotation are labeled as positive, default is False
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      data_augmentation=True)
```
  
**Warning :** This functionnality multiplies the number of instance and can be slow to implement...
  
## Seq2Seq model

To train a seq2seq model to precisely predict the position of an annotation whitin a window, one can set the keyword `seq2seq` to true.
 
```python
from keras_dna import Generator

### Return the precise location of the annotation within every window, default is False
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      seq2seq=True)
```

The labels shape will be `(batch_size, seq_len, nb cell type, nb annotation)`

**Warning :** This functionnality combines with data_augmentation may take a while ...

## Positive definition

In the case of a model predicting several annotations, the same window can be labeled by more than one value. One use the keyword `define_positive` to change the definition of a positive label. It can either be a sequence that contains all the annotation or a sequence that just match a part of the annotation (it does not mean that a data_augmentation will be applied).

```python
from keras_dna import Generator

### Positive label contain all the annotation, default behaviour
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      define_positive='match_all')
                      
### Positive label contain just a part of the annotation
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      define_positive='match_any')
```

## Negative / Positive ratio

One can change the negative / positive ratio by using the keyword `negative_ratio` which is by default set to 1. A positive example contains at least one annotation, a negative sample if far away from any selected annotation.

```python
from keras_dna import Generator

### Two times more negative examples than positive
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      negative_ratio=2)

### Returning all the negative examples
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      negative_ratio='all')
```

## Negative examples

Negative examples can either be ignored, be random sequences or be real DNA sequences away from any selected annotations. Use the keyword `negative_type` to select the desired behaviour.

```python
from keras_dna import Generator

### Returning only positive examples
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      negative_type=None)

### Returning real negative examples (default)
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      negative_type='real')

### Returning only random negative examples
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      negative_type='random')
```

## Using the strand of the example

If the passed annotation files precise the strand of the example of every annotation then by setting the keyword `use_strand` to True, the `Generator` will reverse complement the example in the backward strand. This may be useful for the model to read the DNA sequence like the cell machinery.

```python
from keras_dna import Generator

### DNA sequence in the natural strand
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      use_strand=True)
```
-----------------------------------
  
                      
 
                      
















