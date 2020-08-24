# Keras_dna: simplifying deep genomics

![Keras_dna logo](./docs/favicon.ico)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/etirouthier/keras_dna/LICENSE)

## Description:

Keras_dna is an API that helps quick experimentation in applying deep learning to genomics. It enables quickly feeding a keras model (tensorflow) with genomic data without the need of laborious file conversions or storing tremendous amount of converted data. It reads the most common bioinformatics files and creates generators adapted to the keras models.

Use Keras_dna if you need a library that:

- Allows fast usage of standard bioinformatic data to feed a keras model (nowaday standard for tensorflow).
- Helps formatting the data to the model's needs.
- Facilitates the standard evaluation of a model with genomics data (correlation, AUPRC, AUROC)

Read the documentation at [keras_dna](https://keras-dna.readthedocs.io).

Keras_dna is compatible with: __Python 3.6__.


------------------

## Guiding principles:

- Furnishing a simplified API to create generators of genomic data.

- Reading the DNA sequence directly and effectively from fasta files, to discard the need of conversion.

- Generating the DNA sequence corresponding to the desired annotation (can be sparse annotation or continuous), passed with standard bioinformatic files (gff, bed, bigWig, bedGraph).

- Easily adapting to the type of annotations, their number, the number of different cell types or species.

------------------


## Getting started:

The core classes of keras_dna are `Generator`, to feed the keras model with genomical data, and `ModelWrapper` to attach a keras model to its keras_dna `Generator`.

`Generator` creates batches of DNA sequences corresponding to the desired annotation.

First example, a `Generator` instance that yields DNA sequences corresponding to a given genomical function (here binding site) as the positive class and other sequences as the negative class. The genome is furnished through a fasta file and the annotation is furnished with a gff file (could have been a bed), the DNA is one-hot-encoded, the genomical functions that we want to target need to be passed in a list.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'])
```

Second example, a `Generator` for continuous annotation, this time the annotation is furnished through a bigWig file (it could have been a wig or a bedGraph, but then a file containing the chromosome sizes needs to be passed as size), the desired length of DNA sequences need to be passed. This `Generator` instance yields all the DNA sequences of length 100 in the genome and labels them with the coverage at the nucleotide at the center.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.bw'],
                      window=100)
```
`Generator` owns a lot of keywords to adapt the format of the data both to the keras model and to the task at hand (predicting the sequences' genomical function in different cellular types, classifying between several different functions, predicting from two different inputs, labelling DNA sequences with both their genomical functions and an experimental coverages...)


`ModelWrapper` is a class designed to unify a keras model to its generator in order to simplify further usage (prediction, evaluation) of the model. 

```python
from keras_dna import ModelWrapper, Generator
from tensorflow.keras.models import Sequential()

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.bw'],
                      window=100)
                      
model = Sequential()
### the model need to be compiled
model.compile(loss='mse', optimizer='adam')
 
wrapper = ModelWrapper(model=model,
                       generator_train=generator)
```
 
Train the model with `.train()`
```python
wrapper.train(epochs=10)
```
 
Evaluate the model on a chromosome with `.evaluate()`
```python
wrapper.evaluate(incl_chromosomes=['chr1'])
```

Predict on a chromosome with `.predict()`
```python
wrapper.predict(incl_chromosomes=['chr1'], chrom_size='species.chrom.sizes')
```

Save the wrapper in hdf5 with `.save()`
```python
wrapper.save(path='./path/to/wrapper', save_model=True)
```
 
------------------


## Installation:


**Dependencies**:

- pandas
- numpy
- pybedtools
- pyBigWig
- kipoiseq
- tensorflow 2
              
 We also strongly advice installing [genomelake](https://github.com/kundajelab/genomelake) for fast reading of fasta files. 
 
 - **Install Keras_dna from PyPI:**

Note: These installation steps assume that you are on a Linux or Mac environment.
If you are on Windows, you will need to remove `sudo` to run the commands below.

```sh
sudo pip install keras_dna
```

If you are using a virtualenv, you may want to avoid using sudo:

```sh
pip install keras_dna
```

Note that libcurl (and the `curl-config` command) are required for installation. This is typically already installed on many Linux and OSX systems (this is also easilya vailable using a conda env, in practise we advise installing pyBigWig with conda before installing keras_dna).


- **Alternatively: install Keras_dna from the GitHub source:**

First, clone Keras using `git`:

```sh
git clone https://github.com/etirouthier/keras_dna.git
```

 Then, `cd` to the Keras_dna folder and run the install command:
```sh
cd keras_dna
sudo python setup.py install
```

------------------
