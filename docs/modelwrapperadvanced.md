# ModelWrapper advanced

## AUROC - AUPRC

A standard evaluation of classification model is to calculate the AUROC or the AUPRC. `Generator` enables to easily evaluating this quantity. First, if the model is designed to predict several function on several cell types, they will be treated one by one to evaluate the model.

The standard definition of positive example is to consider one sequence as positive per annotation occurence and as negative example all the sequences of the desired length without any intersection with positive sequences. Other sequences, as they are a mix of positive sequences and the sequences receive no labels and are not taken into account in the calculation.

If the keyword `data_augmentation` is set to True then the positive examples will be all the sequences that contains a whole example of an annotation. The negative examples are the sequences without any intersection with a positive example. (default behaviour)

```python

...

from keras_dna import Generator, ModelWrapper

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.gff'],
                      annotation_list=['binding site', 'enhancer'],
                      incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4'])

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr5'])

wrap.train(epochs=10)

### AUROC curve, default data_augmentation=True
wrap.get_auc(incl_chromosomes=['chr6'])

### AUROC curve, data_augmentation=False
wrap.get_auc(incl_chromosomes=['chr6'], data_augmentation=False)

### AUPRC curve, data_augmentation=False
wrap.get_auc(incl_chromosomes=['chr6'],
             data_augmentation=False,
             curve='PR')
```

It returns a list of dictionary with the cell type index, the name of the annotation and the value of the AUC. The cell type index is the same as in the list pass through `annotation_files`.

We can also evaluate the AUC on another species:

```python

...

from keras_dna import Generator, ModelWrapper

generator = Generator(batch_size=64,
                      fasta_file='species1.fa',
                      annotation_files=['ann1.gff'],
                      annotation_list=['binding site', 'enhancer'],
                      incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4'])

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr5'])

wrap.train(epochs=10)

### Evaluating on another species
wrap.get_auc(incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4'],
             fasta_file='species2.fa',
             annotation_files=['ann2.gff'])
```

For `MultiGenerator` this procedure needs to be followed to specify the species on which one wants to evaluate the AUC.

The number of file in annotation_list should be the same in both the generator and the `.get_auc()` method. If one wants to evaluate on less file one should complete the list with zeros and place the files at the position corresponding to their cell type.

```python

...

from keras_dna import Generator, ModelWrapper

generator = Generator(batch_size=64,
                      fasta_file='species1.fa',
                      annotation_files=['species1_cell1_ann.gff', 'species1_cell2_ann.gff', 'species1_cell3_ann.gff'],
                      annotation_list=['binding site', 'enhancer'],
                      incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4'])

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr5'])

wrap.train(epochs=10)

### Evaluating on another species but only on cell2 and cell3
wrap.get_auc(incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4'],
             fasta_file='species2.fa',
             annotation_files=[0, 'species2_cell2_ann.gff', 'species2_cell3_ann.gff'])
```
## Correlation

Evaluating the correlation is a standard evaluation for continuous data. The method `.get_correlation()` enables easily evaluating the correlation between the predicted and real coverage for every annotation and cell type.

```python

...

from keras_dna import Generator, ModelWrapper

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['cell1_ann1.bw', 'cell1_ann2.bw', 'cell2_ann1.bw', 'cell2_ann2.bw'],
                      nb_annotation_type=2,
                      window=299,
                      incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4'])

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr5'])

wrap.train(epochs=10)

### Correlation for every cell type and annotation
>>> wrap.get_correlation(incl_chromosomes=['chr6'])
{'correlate_0_0' : 0.68, 'correlate_0_1' : 0.62, 'correlate_1_0' : 0.65, 'correlate_1_1' : 0.36}
```
In this example 'correlate_0_0' correspond to the correlation between the predicted coverage of ann1 on cell1 and the truth, 'correlate_0_1' for ann2 on cell1 an so on ...

To calculate the correlation on another species or to evaluate a `MultiGenerator` one can pass a fasta file and its corresponding annotation file with `fasta_file` and `annotation_files`. Fill with zeros to exclude some cell type or annotation from the evaluation in the second species.

```python
...

from keras_dna import Generator, ModelWrapper

generator = Generator(batch_size=64,
                      fasta_file='species1.fa',
                      annotation_files=['s1_cell1_ann1.bw', 's1_cell1_ann2.bw', 's1_cell2_ann1.bw', 's1_cell2_ann2.bw'],
                      nb_annotation_type=2,
                      window=299,
                      incl_chromosomes=['chr1', 'chr2', 'chr3', 'chr4'])

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr5'])

wrap.train(epochs=10)

### Correlation for every cell type 2 in  species2
>>> wrap.get_correlation(incl_chromosomes=['chr6'],
                         fasta_file='species2',
                         annotation_files=[0, 0, 's2_cell2_ann1.bw', 's2_cell2_ann2.bw'])
{'correlate_1_0' : 0.65, 'correlate_1_1' : 0.36}
```

--------------------------------------
