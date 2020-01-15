# Keras_dna: simplifying deep genomics

![Keras_dna logo](/docs/favicon.ico)

[![Build Status](https://travis-ci.org/keras-team/keras.svg?branch=master)](https://travis-ci.org/keras-team/keras)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

## Description:

Keras_dna is an API that help quick experimentation of application of deep learning to genomics. It anables to quickly feed a keras model with genomic data without the need of laborious file convertion and storing tremendous amount of converted data. It reads the most commun bioinformatics file types and create a generator adapted to a keras model.

Use Keras_dna if you need a library that:

- Allows fast usage of standard bioinformatics data to feed a keras model.
- Is able to adapt to the needed format of data.
- Facilitates the standard evaluation of a model with genomics data (correlation, AUPRC, AUROC)

Read the documentation at [](https:).

Keras is compatible with: __Python 3.6__.


------------------

## Guiding principles:

- Fournishing a simplified API to create generator of genomical data.

- Reading the DNA sequence directly and effectively in fasta file to discard the need of storing huge amounts of data.

- Generating the DNA sequence corresponding to the desired annotation (can be sparse annotation or continuous), passed with standard bioinformatic files (gff, bed, bigWig, bedGraph).

- Easily adapt to the type of annotation, their number, the number of different cell type or species.

------------------


## Getting started:

The core data structures of Keras_dna are a __generator__, to feed the keras model with genomical data, and a __modelwrapper__ to unify the keras model to its keras_dna generator.

`Generator` is able to create batch of DNA sequence corresponding to the desired annotation.

First example, a `Generator` that will return the DNA sequence underlying a given sparse annotation (here binding site). The DNA sequence is fournished with a fasta file and the position of annotation is fournished with a gff file (could have been a bed), the DNA is one-hot-encoded, the annotations that we want to target need to be passed in a list.

```python
from keras_gpu import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'])
```

Second example, a `Generator` for continuous annotation, this time the file is a bigWig file (it can also be passed with a wig or a bedGraph, but then a file containing the size of chromosome need to be passed as size), the window need to be passed (as well as the batch size and the fasta file). This generator will generate all the window of length 100 in the DNA and will label it with the coverage at the center nucleotid.

```python
from keras_gpu import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.bw'],
                      window=100)
```
`Generator` owns a lot of keywords to adapt the format of the data to the keras model and to adapt to our task (predicting several annotation at the same time in different cellular type, adding a secondary input, adding a secondary target...)


`ModelWrapper` is a class designed to unify a keras model to its generator so that to simplify further utilisations of the model (prediction, evaluation). 

```python
from keras_gpu import ModelWrapper, Generator
from keras.models import Sequential()

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.bw'],
                      window=100)
                      
 model = Sequential()
 ### the model need to be compiled
 model.compile(loss='mse, optimizer='adam')
 
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
- kipoi
- kipoiseq
- keras
- tensorflow
              
 We also strongly advice to install [genomelake](https://github.com/kundajelab/genomelake) for fast reading in fasta file. 
 
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
 
 
 
 
 
 
 
 
 











