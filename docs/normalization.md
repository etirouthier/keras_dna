# Normalization

## Introduction

As seen in [Generator](./generators.md) and in [MultiGenerator](./multigenerator.md) there is some normalization that could be applied on continuous dataset.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw']
                      window=1000,
                      normalization_mode='max')
```

## Single normalization

Follow the previous example to apply only one normalization to the labels of the model. The available normalization type are :

- `'zscore'`: substracting the mean of the mean of the values and dividing by the standard deviation.
- `'max'`: dividing by the maximum of the values
- `'perctrim'`: trimming the distribution to the last percentile (all values above are set equal to the last percentile)
- `'logtransform'`: taking the log of the values : `np.log(values + 1)`
- `'min_max'`: substracting `min` and dividing by `(max - min)`, the values are between 0 and 1 after.

The statistics are taken on the whole genome (only the chromosomes named with a number) for consistency between all the generagor created with a file.

## Applying two successive normalization

It is usually useful to apply two successive normalization, in particular to trim the sequence before applying the other available normalization. `Generator` owns the possibility to apply up to two successive normalization. In this case the available normalization modes are the same and statistics (min, max, std) are taken on a subsample of the whole genome.

```python
from keras_dna import Generator

### Trimming the distribution then dividing by the maximum
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw']
                      window=1000,
                      normalization_mode=['max', 'perctrim'])

### Log transform before taking the zscore
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw']
                      window=1000,
                      normalization_mode=['zscore', 'logtransform'])

```
