# Generating continuous data

## Introduction

`Generator` can be use to feed a keras model with data that are annotated with a continuous function such as MNase or Chip-seq. The inputs file will then be a bigWig file, it can also be a wig or a bedGraph file but then another file containing the length of chromosome is needed (a conversion will be made using Kent's tools, two columns chrom, size tab separated). `Generator` will detect such files with the suffix .bw, .wig, .bedGraph. The length of the generated sequence needs to be passed with the keyword `window`.

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

The standard behaviour will be to generate all the sequence of window length in the DNA (one-hot-encoded) and to label it with the coverage at the centre nucleotide. The standard shape of DNA sequence is `(batch_size, window, 4)` and the standard labels shape is `(batch_size, tg_window, len(annotation_files))`, here `(64, 1, 1)`

## Model for several cell types and annotation

`Generator` is also able to manage several cell types and annotation in the continuous context. Firstly, in this case one file corresponds to one annotation in one cell type, we use the keyword `nb_annotation_type` to sprecify the number of file one wants to predict. Then the list of file **needs to be organised** in the example.

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
As shown in the last example this organisation is optionnal and without it all the labels are set in the 2nd axis.

## Length of the target

`Generator` anables us to change the length of the labels of a window with the keyword `tg_window`. DNA sequences are labeled with another sequence of length `tg_window` representing the coverage (centered).

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

As previously stated `Generator` can be used to train a seq2seq model, the first sequence being the DNA and the second the coverage, the default behaviour is to return the raw sequence of coverage values at the center of the DNA sequence. Using the keyword `downsampling` makes the `Generator` labelling the DNA sequence with the whole corresponding sequence of values but downsampled (if `tg_window` is smaller than `window` then it should divide `window`, if equal then `downsampling` is useless).

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

**Warning :** note that some normalization values are obtained by subsampling the data for memory and time purpose.

## Overlapping of sequence

In the case of a seq2seq model the keyword `overlapping` anables to keep all the data available (i.e with overlapping targets) or to keep only the data with non overlapping targets.

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
