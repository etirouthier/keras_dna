# Generator 

## Introduction

`Generator` class owns several keywords among which two that are mandatory : `batch_size` and `fasta_file`.

```python
from keras_dna import Generator

generator = Generator(batch_size=64, fasta_file='species.fa', ...)
```
## Adapting the output shape

By default the label shape of `Generator` is in general `(batch_size, len(target), nb cell type, nb annotation` or for a non seq2seq classification model `(batch_size, nb cell type, nb annotation)` but one may want to modify this shape to match the output shape of the model (labels are compared with the output of the model). Pass the desired output shape through a tuple (keyword `output_shape`) to do so. Note that the first number of the tuple is the batch size and should match the `batch_size` argument.

```python
from keras_dna import Generator

### Standard shape is (64, 1, 1) we can change it to (64, 1)
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      output_shape=(64, 1))
```

## Choosing the training set

`Generator` enables to choose the training set by choosing chromosomes that will be part of it. To do so use either the keyword `incl_chromosomes` to pass a list of chromosome to include or use `excl_chromosomes` to pass a list of chromosome to exclude.

```python
from keras_dna import Generator

### Restrincting to chromosome 1
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      incl_chromosome=['chr1'])

### Excluding chromosome 1
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      excl_chromosome=['chr1'])
```

## Ignoring labels

If one wants to ignore the labels and that the `Generator` returns only the DNA sequence, set the keyword `ignore_targets` to True.

```python
from keras_dna import Generator

### Only DNA sequence
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      ignore_targets=True)
```

## DNA sequence in string format

Models inspired from the natural language processing domain use DNA sequence in string format. To return the DNA sequences in string format, set `one-hot-encoding` to false in `Generator`. The keyword `force_upper` force the letter to be uppercase.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      one-hot-encoding=False)

>>> next(generator())[0]
'AaTCtGg ... GCtA'

### Forcing uppercase
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      one-hot-encoding=False,
                      force_upper=True)

>>> next(generator())[0]
'AATCTGG ... GCTA'
```

## Changing the alphabet

The default behaviour of `Generator` is to yield DNA sequence one-hot-encoded with the alphabet 'ACGT' (A=1, C=2,...). This alphabet can be changed with the keyword alphabet.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      one-hot-encoding=True,
                      alphabet='ATGC')
```


## Adding weights to the training

In genomics, most of the data are unbalanced in terms of distribution (for example there are much more background sequences than annotated sequences), adding weights to the training can mitigate this fact. The keyword `weighting_mode` and `bins` will be covered in details in [Weights](weights.md).

## Adapting the shape of the one-hot-encoded DNA sequence

Two keywords are necessary to adapt the shape of the DNA sequence: `alphabet_axis` set the axis encoding for ACTG, and dummy_axis add a dimension to the sequence on the desired axis.

Note that the numerotation of axis does not take the batch axis into account.

```python
from keras_dna import Generator

### Standard input shape is (64, 299, 4)
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299)

### We want to adapt it to a Conv2D, we need (64, 299, 1, 4)
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      dummy_axis=1,
                      alphabet_axis=2)

### Or (64, 299, 4, 1)
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      dummy_axis=2,
                      alphabet_axis=1)                     
```

## Reverse complement DNA sequences

It is sometimes usefull to reverse complement the DNA sequence. `Generator` owns the keyword `rc` to do so.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      rc=True)

```

## Name of chromosomes

The annotation files and the fasta file are sometimes incoherent in there naming of chromosomes. To correct this relatively frequent issue, the keyword `num_chr` can be used, set to True it drop 'chr' from the chromosome name in the annotation file if present, set to False (default) it add 'chr' to the chromosome name in the annotation file if absent.

```python
from keras_dna import Generator

### Fasta file use '1' and anotation_files 'chr1'
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      num_chr=True)

### Fasta file use 'chr1' and anotation_files '1'
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      num_chr=False)

### Fasta file use '1' and anotation_files '1'
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      num_chr=True)
```


## Adding a secondary inputs or labels

`Generator` enables to add a secondary inputs or labels. This secondary inputs is necessarily a continous input and need to be passed with a bigWig file. It consists in the coverage on the interval where the DNA sequence was taken. Several keywords are used to adapt this secondary input to the need (please refer to [Continuous Data](continuous.md) for details, keywords are highly similar):

- `sec_inputs`: list of .bw file to use as secondary input, similar as `annotation_files`.
- `sec_input_length`: the length of the sencondary input, similar to `tg_window` (default is the length of the DNA sequence).
- `sec_input_shape`: the default shape is similar to what appends with a continuous data, this keyword anables to adapt.
- `sec_nb_annotation`: similar as `number_of_annotation`.
- `sec_sampling_mode`: if we want the secondary sequence to cover all the DNA sequence but downsampled, similar to `downsampling`.
- `sec_normalization_mode`: similar to `normalization_mode`
- `use_sec_as`: {'targets', 'inputs'}.

## Anticipating the input-label shape

The main goal of the `Generator` class is to yields data in an adapted format to train a keras model. Use the class methods `predict_input_shape`, `predict_label_shape` and `predict_sec_input_shape` to calculate those shape before creating an instance. Note that the batch size is not included in the returned tuple.

```python
>>> from keras_dna import Generator

>>> Generator.predict_input_shape(batch_size=64,
                                  fasta_file='species.fa',
                                  annotation_files='ann.bw',
                                  window=299,
                                  output_shape=(64, 1))
(299, 4)

>>> Generator.predict_label_shape(batch_size=64,
                                  fasta_file='species.fa',
                                  annotation_files='ann.bw',
                                  window=299,
                                  output_shape=(64, 1))
(1,)
```


-------------------------------------------------
