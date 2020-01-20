# Normalization

## Introduction

As seen in [Generator](./generators.md) and in [MultiGenerator](./multigenerators.md) there is some normalization that could be applied on continuous dataset.

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
- `'max'`: dividing by the macimum of the values
- `'perctrim'`: trimming the distribution to the last percentile (all values above are set equal to the last percentile)
- `'logtransform'`: taking the log of the values `np.log(values + 1)`
- `'min_max'`: substracting the min and dividing by `(max - min)`, the values are between 0 and 1 after.
