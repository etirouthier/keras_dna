# Regression

## Introduction

`Generator` can be use to feed a keras model with sequences that are annotated with a continuous function such as MNase or Chip-seq. The inputs file will then be a bigWig, a wig or a bedGraph file. `Generator` will detect such files with the suffix .bw, .wig, .bedGraph. The length of the generated sequence needs to be passed with the keyword `window`.

Note that for wig and bedGraph file the chromosome size is needed.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299)

### With a wig file
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.wig'],
                      size='species.chrom.sizes',
                      window=299)
```

The standard behaviour will be to generate all the DNA subsequence of length equal to window (one-hot-encoded) and to label it with the coverage at the centre nucleotide. The standard shape of DNA subsequence is `(batch_size, window, 4)` and the standard labels shape is `(batch_size, tg_window, len(annotation_files))`, here `(64, 1, 1)`

## Regression on several cell types and/or annotations

`Generator` is also enables to perform multiple regression on several cell types at the same time. In this case one argument file corresponds to one annotation in one cell type. Use the keyword `nb_annotation_type` to specify the number of regression one wants to perform on every cell type. Then the list of file **needs to be organised** as in the example.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['cell1_ann1.bw', 'cell1_ann2.bw', 'cell2_ann1.bw', 'cell2_ann2.bw'],
                      nb_annotation_type=2,
                      window=299)

### shape of the labels
>>> next(generator())[1].shape
(64, 1, 2, 2)

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['cell1_ann1.bw', 'cell1_ann2.bw', 'cell1_ann3.bw'],
                      nb_annotation_type=3,
                      window=299)

### shape of the labels
>>> next(generator())[1].shape
(64, 1, 1, 3)

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['cell1_ann1.bw', 'cell1_ann2.bw', 'cell1_ann3.bw'],
                      window=299)

### shape of the labels
>>> next(generator())[1].shape
(64, 1, 3)
```                      
As shown in the last example this organisation is optionnal, without it all the labels are set in the 2nd axis.

## Length of the labels

`Generator` enables us to change the length of the labels corresponding to a DNA window with the keyword `tg_window`. DNA sequences are labeled with a sequence of number of length `tg_window` representing the coverage (centered).

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299,
                      tg_window=100)

### shape of the labels
>>> next(generator())[1].shape
(64, 100, 1)
```

## Downsampling

As previously stated `Generator` can be used to train a seq2seq model, the first sequence being a DNA window and the second the coverage, the default behaviour is to return the raw sequence of coverage values at the center of the DNA window. Using the keyword `downsampling` the `Generator` labels the DNA window with the whole corresponding sequence of values but downsampled (if `tg_window` is smaller than `window` then it should divide `window`, if equal then `downsampling` is useless).

```python
from keras_dna import Generator

### Downsampling by taking the mean of values (here of 3 values)
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=300,
                      tg_window=100,
                      downsampling='mean')

### Downsampling by taking one values amoung several (here 3)
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=300,
                      tg_window=100,
                      downsampling='downsampling')
```

## Normalizing the data

Continuous biological data needs usually to be normalized. This is possible by using the keyword `normalization_mode`. The different kind of available normalization will be discussed in another section.

```python
from keras_dna import Generator

### Cutting the distribution to the last percentile
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299,
                      normalization_mode='perctrim')
                      
### Cutting then dividing by the max
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299,
                      normalization_mode=['max', 'perctrim'])
```

**Warning :** note that some normalization parameters are obtained by subsampling the data for memory and time purpose.

## Overlapping of sequences

In the case of a seq2seq model the keyword `overlapping` enables to generate either all the data available (i.e with overlapping targets) or to generate only the data with non overlapping targets.

```python
from keras_dna import Generator

### Default behaviour
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299,
                      overlapping=True)
                      
### Discarting sequence with overlapping targets
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299,
                      overlapping=False)
```

------------------------
