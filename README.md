# Keras_dna: simplifying deep genomics

![Keras_dna logo](/docs/favicon.ico)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/etirouthier/keras_dna/LICENSE)

## Description:

Keras_dna is an API that helps quick experimentation in applying deep learning to genomics. It enables to quickly feed a keras model (tensorflow) with genomic data without the need of laborious file convertion and storing tremendous amount of converted data. It reads the most commun bioinformatics file types and create a generator adapted to a keras model.

Use Keras_dna if you need a library that:

- Allows fast usage of standard bioinformatics data to feed a keras model (nowaday standard for tensorflow).
- Is able to adapt to the needed format of data.
- Facilitates the standard evaluation of a model with genomics data (correlation, AUPRC, AUROC)

Read the documentation at [keras_dna](https://keras-dna.readthedocs.io).

Keras_dna is compatible with: __Python 3.6__.


------------------

## Guiding principles:

- Fournishing a simplified API to create generator of genomical data.

- Reading the DNA sequence directly and effectively in fasta file to discard the need of storing huge amounts of data.

- Generating the DNA sequence corresponding to the desired annotation (can be sparse annotation or continuous), passed with standard bioinformatic files (gff, bed, bigWig, bedGraph).

- Easily adapt to the type of annotation, their number, the number of different cell type or species.

------------------


## Getting started:

The core data structures of Keras_dna are a __generator__, to feed the keras model with genomical data, and a __modelwrapper__ to attach a keras model to its keras_dna generator.

`Generator` is able to create batch of DNA sequence corresponding to the desired annotation.

First example, a `Generator` that will return DNA subsequences corresponding to a given function (here binding site) as positive class and subsequences far away as negative class. The DNA sequence is fournished through a fasta file and the annotation is fournished with a gff file (could have been a bed), the DNA is one-hot-encoded, the function names that we want to target need to be passed in a list.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'])
```

Second example, a `Generator` for continuous annotation, this time the file is a bigWig file (it can also be passed with a wig or a bedGraph, but then a file containing the size of chromosome need to be passed as size), the length of desired window need to be passed. This generator will generate all the window of length 100 in the DNA and will label it with the coverage at the center nucleotid.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.bw'],
                      window=100)
```
`Generator` owns a lot of keywords to adapt the format of the data both to the keras model and to our task (predicting the sequence function in different cellular type, choosing between several different functions, adding a secondary input, adding a secondary target...)


`ModelWrapper` is a class designed to unify a keras model to its generator  in order to simplify further usage of the model (prediction, evaluation). 

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

Note that libcurl (and the `curl-config` command) are required for installation. This is typically already installed on many Linux and OSX systems (this is also available easily if using a conda env).


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
 
 
 
 
 
 
 
 
 











