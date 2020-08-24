# Model Wrapper basics


## Introduction

`ModelWrapper` is a class designed to attach the keras model to its corresponding keras_dna `Generator`. It is especially useful to train, evaluate or make predictions with the model. The core arguments are:

- `model`: a compiled keras model
- `generator_train`: a keras_dna generator restricted to the chromosomes one wants to train on.
- `generator_val`: a keras_dna generator restricted to the chromosomes one wants to validate on. It must return data in the same shape as `generator_train`.
- `validation_chr`: if `generator_val` is identical to `generator_train` except for the chromosomes included then just pass the validation chromosome with this keyword (no need to pass a `generator_val`).
- `weights_val`: set it to True if one wants to keep the same weighting mode in the `generator_val` as in `generator_train`, to False to disable weighting (default is False).

Creating a `ModelWrapper`:

```python
from keras_dna import Generator, ModelWrapper
from tensorflow.keras import Sequential

model = Sequential()
model.compile(loss='mse', optimizer='adam')

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299,
                      incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4', 'chr5'],
                      weighting_mode='balanced')

### Not including weights in validation (default)
wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr6', 'chr7'],
                    weights_val=False)
```

When creating a `ModelWrapper` with generator_train being a `MultiGenerator` instance, the chromosome names may vary between species involved in the generator, then it is mandatory to create and pass a generator_val:

```python
from keras_dna import Generator, MultiGenerator, SeqIntervalDl
from tensorflow.keras import Sequential

model = Sequential()
model.compile(loss='mse', optimizer='adam')

dataset1_train = SeqIntervalDl(fasta_file='species1.fa',
                               annotation_files=['ann1.bw'],
                               window=299,
                               incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4', 'chr5'])

dataset2_train = SeqIntervalDl(fasta_file='species2.fa',
                               annotation_files=['ann2.bw'],
                               window=299,
                               incl_chromosomes=['chr1', 'chr2', 'chr3'])
                               
dataset1_val = SeqIntervalDl(fasta_file='species1.fa',
                               annotation_files=['ann1.bw'],
                               window=299,
                               incl_chromosomes=['chr6', 'chr7'])

dataset2_val = SeqIntervalDl(fasta_file='species2.fa',
                               annotation_files=['ann2.bw'],
                               window=299,
                               incl_chromosomes=['chr4'])

generator_train = MultiGenerator(batch_size=64, dataset_list=[dataset1_train, dataset2_train])
generator_val = MultiGenerator(batch_size=64, dataset_list=[dataset1_val, dataset2_val])

wrap = ModelWrapper(model=model,
                    generator_train=generator_train,
                    generator_val=generator_val)
```

## Training

With the `ModelWrapper` instance in hand one can easily train the model with `.train()`, the only mandatory keyword is `epochs` to specify the number of epochs. One can also pass `steps_per_epoch` and `validation_steps`  but also all the available options accepted by the method `.fit_generator()` of a keras model.

```python
...
wrap = ModelWrapper(model=model,
                    generator_train=generator_train,
                    generator_val=generator_val)
                    
wrap.train(epochs=10)

### Adding keras options
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

checkpointer = ModelCheckpoint(filepath=path_to_output_file,
                                   monitor='val_loss',
                                   verbose=0, 
                                   save_best_only=True, 
                                   save_weights_only=False, 
                                   mode='min',
                                   period=1)

early = EarlyStopping(monitor='val_loss',
                      min_delta=0,
                      patience=5,
                      verbose=0,
                      mode='auto'
tensorboard = TensorBoard(log_dir=path_to_tensorboard, update_freq=200)

wrap.train(steps_per_epoch=500, 
           epochs=100,
           validation_steps=200,
           callbacks=[checkpointer, early, tensorboard])
```

## Evaluating

To evaluate the model on the desired chromosomes, use `.evaluate()`. If generator_train is a `Generator` instance one needs to specify the chromosomes with the keyword `incl_chromosomes`. If generator_train is a `MultiGenerator` instance one needs to create a full generator and pass it with `generator_eval`. One can also pass keywords corresponding to the keras model method `.evaluate_generator()`.


Evaluation of a `Generator`:
```python

...

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299,
                      incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4', 'chr5'])

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr6', 'chr7'])


wrap.evaluate(incl_chromosomes=['chr8'])
```

Evaluation of a `MultiGenerator`:
```python

...

dataset1_train = SeqIntervalDl(fasta_file='species1.fa',
                               annotation_files=['ann1.bw'],
                               window=299,
                               incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4', 'chr5'])

dataset2_train = SeqIntervalDl(fasta_file='species2.fa',
                               annotation_files=['ann2.bw'],
                               window=299,
                               incl_chromosomes=['chr1', 'chr2', 'chr3'])
                               
dataset1_val = SeqIntervalDl(fasta_file='species1.fa',
                               annotation_files=['ann1.bw'],
                               window=299,
                               incl_chromosomes=['chr6', 'chr7'])

dataset2_val = SeqIntervalDl(fasta_file='species2.fa',
                               annotation_files=['ann2.bw'],
                               window=299,
                               incl_chromosomes=['chr4'])

generator_train = MultiGenerator(batch_size=64, dataset_list=[dataset1_train, dataset2_train])
generator_val = MultiGenerator(batch_size=64, dataset_list=[dataset1_val, dataset2_val])

wrap = ModelWrapper(model=model,
                    generator_train=generator_train,
                    generator_val=generator_val)
                    
dataset1_eval = SeqIntervalDl(fasta_file='species1.fa',
                              annotation_files=['ann1.bw'],
                              window=299,
                              incl_chromosomes=['chr8', 'chr9'])

dataset2_eval = SeqIntervalDl(fasta_file='species2.fa',
                              annotation_files=['ann2.bw'],
                              window=299,
                              incl_chromosomes=['chr5'])

generator_eval = MultiGenerator(batch_size=64, dataset_list=[dataset1_eval, dataset2_eval])

wrap.evaluate(generator_eval=generator_eval)
```

## Predicting

Use the `.predict()` to make predictions with a `ModelWrapper`. One can choose the chromosomes on which to predict by specifying them with `incl_chromosomes` and by passing a file containing the chromosome length (in two tab separated columns, suffix must be .chrom.sizes) in `chrom_size`. 

One can also predict on another species by passing a fasta file to `fasta_file`, with the corresponding `chrom_size`. This option is mandatory in the case of a `MultiGenerator`. Predictions are saved if `export_to_path` is specified (one file per annotation in one cell type, the format is bigWig).

```python
...

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr6', 'chr7'])

### Predict on the same species
wrap.predict(incl_chromosomes=['chr8', 'chr9'],
             chrom_size='species.chrom.sizes')
             
### Predict on another species and exporting to bigWig
wrap.predict(incl_chromosomes=['chr1', 'chr2'],
             chrom_size='species2.chrom.sizes',
             fasta_file='species2.fa',
             export_to_path='path/to/species2')
```

**Note :** predictions are made on all the available data in the specified chromosome even for sparse data, in this case it displays the probability of a nucleotid to have a given function.

If one wants to predict only on a region of a chromosome, one need to specify the begining and the end of the region by passing a tuple to the keyword `start_stop`. One needs to pass a list of tuples, one tuple per chromosome included.


```python
...

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr6', 'chr7'])

### Predict on a region of chromosome 8 and another region of chromosome 9
wrap.predict(incl_chromosomes=['chr8', 'chr9'],
             start_stop=[(10000, 20000), (50000, 55000)],
             chrom_size='species.chrom.sizes')

```

Finally, the prediction can be made on the other strand of the DNA. One only needs to set the keyword `rc` to True.

```python
...

### Predict on the reverse complemented sequence
wrap.predict(incl_chromosomes=['chr8', 'chr9'],
             rc=True,
             chrom_size='species.chrom.sizes')

```

## Saving

To save a `ModelWrapper` use the method `.save()` with a path as argument. It creates a hdf5 file, the keras model is saved as usual and a dictionary describing how to reconstruct the `Generator` is saved as well.

Be aware of one subtlety, a usual keras callback is `ModelCheckpoint` that enables saving the best model obtained during the training; but the model continues to train after reaching its best. By saving the model after the training with the method `.save()` the best model will be overwritten by the last obtained. To avoid this set the keyword `save_model` to False (default behaviour).

```python
...

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr6', 'chr7'])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath=path_to_output_file,
                               monitor='val_loss',
                               verbose=0, 
                               save_best_only=True, 
                               save_weights_only=False, 
                               mode='min',
                               period=1)

wrap.train(steps_per_epoch=500, 
           epochs=100,
           validation_steps=200,
           callbacks=[checkpointer])

### The default behaviour does not overwritte the saved model
wrap.save(path=path_to_output_file)

wrap.train(epochs=100)

### To save the model one needs to specify save_model=True
wrap.save(path=path_to_output_file,
          save_model=True)
```

## Loading

Loading a `ModelWrapper` consists in loading the keras model and reconstructing the same `Generator`.

```python
from keras_dna.model import load_wrapper

wrapper = load_wrapper(path_to_model)
```

To reconstruct the model the data need to be present and organised as they were passed in `Generator`, so it limits the shareability of the model. The function `load_generator_command` returns a dictionary with the command needed to recreate both the train and validation generators. The keras model needs to be loaded separately.

Note: it can be used to retrain a model on new data, just change the input files.

```python
from keras_dna.model import load_generator_command

dict = load_generator_command(path_to_model)

### Class of generator_train:
### either a Generator instance
>>> dict['type']
'Generator'

### or a MultiGenerator with SeqIntervalDl dataset
>>> dict['type']
'MultiSeq'

### or a MultiGenerator with StringSeqIntervalDl dataset
>>> dict['type']
'MultiStringSeq'

### Access the command dictionary (or list in the case of a MultiGenerator, one per dataset)
>>> dict['arguments']
{'fasta_file' : 'species.fa',
 'batch_size' : 64,
 ...}
```

------------------------
