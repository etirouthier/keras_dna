# Dataset

## Introduction

Dataset is used by `Generator` to process the data, in the particular case of a `MultiGenerator` (see [MultiGenerator](multigenerator.md)) one needs to use them directly and this section shows how.

There are two classes of dataset : 

- `SeqIntervalDl`: used to generate batches of one-hot-encoded DNA sequences.
- `StringSeqIntervalDl`: used to generate batches of DNA sequences as strings.

## SeqIntervalDl

The `SeqIntervalDl` owns the same keywords as the `Generator` class except `batch_size`, `one-hot-encoding`, `output_shape`, `weighting_mode` and `bins`.


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

The DNA sequence is one-hot-encoded and the shape is adaptable by using `dummy_axis` and `alphabet_axis` (see [Generator](generators.md)).

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

**Warning :** the default alphabet use [genomelake](https://github.com/kundajelab/genomelake) leads to an increase of performance.

The type of the np.ndarray created can be chosen using the keyword `dtype`.

## Anticipating the input / label shape

Use the class methods `predict_input_shape`, `predict_label_shape` and `predict_sec_input_shape` to calculate the corresponding shape before creating an instance of a `SeqIntervalDl`. Note that the batch size is not included in the returned tuple.

```python
>>> from keras_dna import SeqIntervalDl

>>> SeqIntervalDl.predict_input_shape(batch_size=64,
                                      fasta_file='species.fa',
                                      annotation_files='ann.bw',
                                      window=299,
                                      output_shape=(64, 1))
(299, 4)

>>> SeqIntervalDl.predict_label_shape(batch_size=64,
                                      fasta_file='species.fa',
                                      annotation_files='ann.bw',
                                      window=299,
                                      output_shape=(64, 1))
(1,)

>>> SeqIntervalDl.predict_sec_input_shape(batch_size=64,
                                          fasta_file='species.fa',
                                          annotation_files='ann.bw',
                                          window=299,
                                          output_shape=(64, 1),
                                          sec_inputs=['ann2.bw', 'ann3.bw'],
                                          sec_input_length=199)
(199, 2)
```


-------------------------------------------------



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

Use the class method `predict_label_shape` and `predict_sec_input_shape` to predict the corresponding shape before creating an instance.

--------------------------------
