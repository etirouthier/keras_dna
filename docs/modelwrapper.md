# Model Wrapper basics

## Introduction

`ModelWrapper` is a class designed to unify the keras model to its corresponding keras_dna generator. It is especially useful to train, evaluate the model or predict with the model. The basics arguments are:

- `model`: compiled keras model
- `generator_train`: a keras_dna generator restricted to the chromosomes one wants to train on.
- `generator_val`: a keras_dna generator restricted to the chromosomes one wants to validate on. It must return the same shape as `generator_train`.
- `validation_chr`: if `generator_val`is absolutely identical to `generator_train` except for the chromosomes included then just pass the validation chromosome with this keyword (no need to pass a `generator_val`).
- `weights_val`: set it to True if one wants to keep the same weighting mode in the `generator_val` as in `generator_train`, to False to disable weighting (default is False).

Creating a `ModelWrapper`:

```python
from keras_dna import Generator, ModelWrapper
from keras import Sequential

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

Creating a `ModelWrapper` for a `MultiGenerator`, the chromosome name may not be the same for every species included so to create a model wrapper with a `MultiGenerator` one needs to create the generator_val before:

```python
from keras_dna import Generator, MultiGenerator, SeqIntervalDl
from keras import Sequential

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

## Training a model

Once the model wrapper is designed we can easily train the model with `.train()`, the only mandatory keyword is `epochs` to specify the number of epochs. One can also pass `steps_per_epoch` and `validation_steps` to specify but also pass all the available option accepted by the module `.fit_generator()` of a keras model.

```python
...
wrap = wrap = ModelWrapper(model=model,
                    generator_train=generator_train,
                    generator_val=generator_val)
                    
wrap.train(epochs=10)

### Adding keras options
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

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




























