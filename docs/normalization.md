# Normalization

## Introduction

As seen in [Generator](./generators.md) and in [MultiGenerator](./multigenerator.md) some normalizations could be applied for regression problem.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw']
                      window=1000,
                      normalization_mode='max')
```

## Single normalization

Follow the previous example to apply only one normalization to the labels. The available normalizations are :

- `'zscore'`: substracting the mean values and dividing by the standard deviation.
- `'max'`: dividing by the maximal values
- `'perctrim'`: trimming the distribution to the last percentile (all values above are set equal to the last percentile)
- `'logtransform'`: taking the log of the values : `np.log(values + 1)`
- `'min_max'`: substracting `min` and dividing by `(max - min)` to clip values between 0 and 1.

The statistics are taken on the whole genome for consistency between all the generator created with a given file. Note that only chromosomes named with a number (arabic or roman) are taken into account.

## Applying two successive normalizations

It is sometimes useful to apply two successive normalizations, in particular trimming the sequence before applying another available normalization. `Generator` owns the possibility to apply up to two successive normalizations. In this case the available normalization are the same but the statistics (min, max, std) are taken on a subsample of the whole genome.

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

-------------------------
