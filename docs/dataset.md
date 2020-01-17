# Dataset

## Introduction

Dataset are used by `Generator` to process the data, in the particular case of a `MultiGenerator` (see [MultiGenerator](multigenerator.md)) one need to use them directly and this section shows how to use them.

There is to class of dataset : 

- `SeqIntervalDl`: used to generate batch of one-hot-encoded DNA sequence.
- `StringSeqIntervalDl`: used to generate batch of string DNA sequence.

## SeqIntervalDl

The `SeqIntervalDl` owns the same keyword as the `Generator` class except `batch_size`, `one-hot-encoding`, `output_shape`, `weighting_mode` and `bins`.


Creating a `SeqIntervalDl` instance:
```python
from keras_dna import SeqIntervalDl

### Minimal dataset for sparse data
dataset = SeqIntervalDl(fasta_file='species.fa',
                        annotation_files=['ann.gff'],
                        annotation_list=['binding site'])
                        
### Minimal dataset for continuous data
dataset = SeqIntervalDl(fasta_file='species.fa',
                        annotation_files=['ann.bw'],
                        window=299)
```

The DNA sequence is one-hot-encoded and we can adapt the shape of this sequence using `dummy_axis` and `alphabet_axis` (see [Generator](generators.md)).

```python
from keras_dna import SeqIntervalDl

dataset = SeqIntervalDl(fasta_file='species.fa',
                        annotation_files=['ann.bw'],
                        window=299)

>>> dataset[0]['inputs'].shape
(1, 299, 4)

dataset = SeqIntervalDl(fasta_file='species.fa',
                        annotation_files=['ann.bw'],
                        window=299,
                        dummy_axis=1,
                        alphabet_axis=2)

>>> dataset[0]['inputs'].shape
(1, 299, 1, 4)

dataset = SeqIntervalDl(fasta_file='species.fa',
                        annotation_files=['ann.bw'],
                        window=299,
                        dummy_axis=2,
                        alphabet_axis=1)

>>> dataset[0]['inputs'].shape
(1, 299, 4, 1)
```

The default alphabet is A, C, G, T (index 0, 1, 2, 3 in the alphabet axis). To change this behaviour use the keyword `alphabet`

```python
from keras_dna import SeqIntervalDl

dataset = SeqIntervalDl(fasta_file='species.fa',
                        annotation_files=['ann.bw'],
                        window=299,
                        alphabet='AGCT')
```

**Warning :** the default alphabet make the use of [genomelake](https://github.com/kundajelab/genomelake) possible leading to an increase of performance.

The type of the np.ndarray created can be chosen using the keyword `dtype`.


## StringSeqIntervalDl


The `StringSeqIntervalDl` owns the same keyword as the `SeqIntervalDl` class except `dummy_axis`, `alphabet_axis`, `alphabet` and `dtype`. 

Creating a `StringSeqIntervalDl` instance:
```python
from keras_dna import StringSeqIntervalDl

### Minimal dataset for sparse data
dataset = StringSeqIntervalDl(fasta_file='species.fa',
                        annotation_files=['ann.gff'],
                        annotation_list=['binding site'])

### Minimal dataset for continuous data
dataset = StringSeqIntervalDl(fasta_file='species.fa',
                              annotation_files=['ann.bw'],
                              window=299)
```

--------------------------------
