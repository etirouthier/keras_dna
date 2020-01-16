# Generator 

## Introduction

`Generator` owns several commun keyword and especially two that are mandatory : `batch_size` and `fasta_file`.

```python
from keras_dna import Generator

generator = Generator(batch_size=64, fasta_file='species.fa', ...)
```
## Adapting the output shape

By default the label shape of `Generator` is generally `(batch_size, len(target), nb cell type, nb annotation` or for a non seq2seq sparse model `(batch_size, nb cell type, nb annotation)` but we may want to modify this shape to adapt to our need. Use the keyword `output_shape` to do so.

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

`Generator` anables to choose the training set by choosing the chromosome that will be part of it. To do so use either the keyword `incl_chromosomes` to pass a list of chromosome to include or use `excl_chromosomes` to pass a list of chromosome to exclude.

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

If we want to ignore the labels and that the `Generator` only returns the DNA sequence we set the keyword `ignore_targets` to True.

```python
from keras_dna import Generator

### Only DNA sequence
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      ignore_targets=True)
```

## Using string DNA sequence

Model inspired from the natural language processing domain can lead to the need of using string DNA sequence to train the model. To return the DNA sequence in string format, set `one-hot-encoding` to false in `Generator`. Set the keyword `force_upper` to True force the letter to be uppercase.

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

## Adding weights to the training

To handle highly unbalanced distributions such as in genomics, the need of adding weights to the training often appears. The keyword `weighting_mode` and `bins` will be covered in details in [Weights](weights.md).

## Adapting the shape of the one-hot-encoded DNA sequence

Two keyword are necessary to adapt the shape of the DNA sequence: `alphabet_axis` that set the axis that will encode for ACTG, and dummy_axis if we want to add an axis will shape 1.

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

It is sometimes usefull to reverse complement the DNA sequence. `Generator` owns the keyword `rc` to to so.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files='ann.bw',
                      window=299,
                      rc=True)

```

## Name of chromosomes

The annotation files and the fasta file are sometimes incoherent in there naming of chromosome. To correct a relatively frequent issue, the keyword `num_chr` can be useful, set to True it drop 'chr' from the annotation file chromosome name if present, set to False (default) it add 'chr' to the annotation file chromosome name if absent.

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

`Generator` anables to add a secondary inputs or labels. This secondary inputs is necessarily a continous input and need to be passed with a bigWig file. It consists in the coverage on the interval where the DNA sequence was taken. Several keywords are used to adapt this secondary input to the needs (please refer to [Continuous Data](continuous.md) for details on the similar keywords):

- `sec_inputs`: list of .bw file to use as secondary input, similar as `annotation_files`.
- `sec_input_length`: the length of the sencondary input, similar to `tg_window` (default is the length of the DNA sequence).
- `sec_input_shape`: the default shape is similar to what appends with a continuous data, this keyword anables to adapt.
- `sec_nb_annotation`: similar as `number_of_annotation`.
- `sec_sampling_mode`: if we want the secondary sequence to cover all the DNA sequence but downsampled, similar to `downsampling`.
- `sec_normalization_mode`: similar to `normalization_mode`
- `use_sec_as`: {'targets', 'inputs'}.


-------------------------------------------------
