# Regression

## Introduction

`Generator` can be used to feed a keras model with DNA sequences annotated by a continuous function such as MNase or ChIP-seq. The corresponding annotation files are formatted as bigWig, wig or bedGraph files. `Generator` will detect such files with the suffix .bw, .wig, .bedGraph. The length of the generated sequence needs to be passed with the keyword `window`.

Note that for wig and bedGraph a file containing the chromosome size in two tab separated columns need to be passed and named *.chrom.sizes.

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

The standard behaviour will be to generate all the DNA sequences of length window (one-hot-encoded) and to label it with the coverage at the centre nucleotide. The standard shape of DNA input sequences is `(batch_size, window, 4)` and the standard labels shape is `(batch_size, tg_window, len(annotation_files))`, here `(64, 1, 1)`

## Regression on several cell types and/or annotations

`Generator` is also able to perform multiple regression on several cell types at the same time. In this case one must pass one argument file for one annotation in one cell type. Use the keyword `nb_annotation_type` to specify the number of annotations one wants to predict on each cell type. Then the list of file **needs to be organised** as in the example.

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
As shown in the last example, without specifying `nb_of_annotation`, all the labels are set in the 2nd axis.

## Length of the labels

The length of the labels can be adapted with the keyword `tg_window`. A DNA sequence is labeled by a sequence of length `tg_window` representing the coverage on its n = `tg_window` center nucleotides.

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

As previously stated, `Generator` can be used to train a seq2seq model, the first sequence being a DNA window and the second the coverage, the default behaviour is to return the raw sequence of coverage values at the center of the DNA window. Using the keyword `downsampling` the `Generator` labels the DNA window with the  corresponding sequence of coverage values but downsampled (if `tg_window` is smaller than `window` then it should divide `window`, if equal then `downsampling` is useless).

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

Continuous biological data need usually to be normalized. This is possible by using the keyword `normalization_mode`. The available normalizations will be discussed in another section.

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

**Warning :** note that some normalization parameters are obtained by subsampling the data for memory and time purposes.

## Overlapping of sequences

In the case of a seq2seq model the keyword `overlapping` enables generating either all the data available (i.e with overlapping targets), or generating only the data with non overlapping labels.

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
