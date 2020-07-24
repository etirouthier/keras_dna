# Classification


## Introduction

`Generator` can be used to feed a keras model with sequences of DNA owning a given genomical function. The position of genomical function on the genome needs to be passed in a bed or gff file and the names of the functions one aims to predict in a list. `Generator` detects suffixes .bed, .gff, .gtf, .gff3 and yields the DNA sequence at the positions of the desired genomical functions.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'])
```

The default behaviour is to generate the DNA sequences at the position of the desired function one-hot-encoded and with a length corresponding to the maximal length encountered (one window per genomical function occurence). It simultaneously yields the same amounts of background DNA sequences as negative example. 

The one-hot-encoded DNA batches have the shape `(batch_size, max length, 4)`. The labels are 0 or 1 with the shape `(batch_size, nb anotation files, len(annotation_list))`, i.e here `(64, 1, 1)`.


## Model for several cell types and annotations

`Generator` can be used to classify between different functions in different cell types. The genome is passed through a fasta file and the positions of genomical functions are given through one gff file per cell type. The gff format is needed to provide the position of several different functions in one file, bed format is not adapted to do so.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['cell_type_1.gff', 'cell_type_2.gff'],
                      annotation_list=['ann1', 'ann2'])
```

The one-hot-encoded DNA batches have the shape `(batch_size, max length, 4)`. The labels are 0 or 1 with the shape `(batch_size, nb anotation files, len(annotation_list))`, i.e here `(64, 2, 2)`.


## Changing input length

To length of inputs can be change with the keyword `seq_len`.

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
 
Usually, the number of backgroud sequence in a genome is far greater than the number of occurence of a given genomical function. This fact leads to unbalanced dataset with fewer positive exemples if one keeps the natural distribution. To mitigate this fact the number of positive exemples can be multiplied by a data augmentation procedure: all the windows containing a whole function occurence are labeled as positive (if the input window is longer than a function occurence  several window are generated per occurence). This functionnality is available through the keyword `data_augmentation`:
 
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

To train a seq2seq model to precisely predict the position of a genomical function occurence whitin a DNA sequence, one can set the keyword `seq2seq` to true.
 
```python
from keras_dna import Generator

### Return the precise location of the annotation within every window, default is False
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['gene'],
                      seq2seq=True)
```

The labels shape is `(batch_size, seq_len, nb cell type, nb annotation)`

**Warning :** This functionnality combines with data_augmentation may take a while ...

## Positive definition

In the case of a model classifying between several functions, the same window can be positively labeled for more than one function. Use the keyword `define_positive` to change the definition of a positive label: either a positive sequence needs to contain a function occurence as a whole or just any part of a function occurence. Note that it does not mean that a data_augmentation will be applied.

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

One can change the negative / positive ratio by using the keyword `negative_ratio` which is by default set to 1. A positive example contains at least one occurence of one of the desired function while a negative sample if far away from any of the desired function occurences.

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

Negative examples can either be ignored, be random sequences or be real DNA sequences away from any of the desired functions occurences. Use the keyword `negative_type` to select the desired behaviour.

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

If the annotation files precise the strand for every genomical function then by setting the keyword `use_strand` to True, the `Generator` will reverse complement the example known to be in the backward strand. This may be useful for the model to read the DNA sequence as the cell machinery.

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
  
                      
 
                      
















