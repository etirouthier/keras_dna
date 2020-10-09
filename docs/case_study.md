# Case studies

## Tutorial

A [Tutorial](https://github.com/etirouthier/keras_dna/blob/master/Tutorial.ipynb) with some toy examples is available to learn how to use keras_dna.

## Negative to positive ratio - cross-species prediction

[Khodabandelou, Routhier, and Mozziconacci](https://peerj.com/articles/cs-278/?utm_source=TrendMD&utm_campaign=PeerJ_TrendMD_0&utm_medium=TrendMD) investigated the influence of the negative to positive example ratio in training sets. They used as a case study genes start sites in the human genome, as well as a cross species context in which they make predictions in a different species from the species used for training.
One needs to pass the human genome FASTA file and a GFF file with gene positions as inputs.
One needs also to specify that the target annotation is gene (`annotation_list` keyword), that the target is only the beginning (`predict` keyword),
that the length of input sequences is 299 bp (`seq_len`) and to exclude chromosome 21 from training (`excl_chromosomes`).
The negative to positive ratio is specified using `negative_ratio`.
After associating the generator and the model in a `ModelWrapper` instance it is easy to train the model  and predict all along the chromosome 21.
By passing the genome of another species through a FASTA file to the predict method it is now possible to predict the gene start sites on another, albeit related, species.

```python
from keras_dna import Generator, ModelWrapper

### Suppose we have a function to create and compile the model
model = create_model()

### Suppose the genes data are stucked in a gff file
generator = Generator(batch_size=64,
                      fasta_file='hg38.fa',
                      annotation_files=['RefGene_hg38.gff'],
                      seq_len=299,
                      annotation_list=['gene'],
                      predict='start',
                      negative_ratio=1, #10, 20, 30, 100
                      excl_chromosomes=['chr19', 'chr20', 'chr21'])
                      
wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr19', 'chr20'])
                    
wrap.train(epochs=20)

### Predict the gene start site all along the chromosome 21
wrap.predict(incl_chromosomes=['chr21'],
             chrom_size='hg38.chrom.sizes',
             export_to_path='hg38_TSS.bw')

### Predict the gene start site on the mouse genome chromosome 19
wrap.predict(incl_chromosomes=['chr19'],
             chrom_size='mm10.chrom.sizes',
             fasta_file='mm10.fa',
             export_to_path='mm10_TSS.bw')
```


### Data-augmentation - seq2seq prediction
[Salekin,  Zhang,  and  Y.  Huang](https://academic.oup.com/bioinformatics/article/34/20/3446/4994793)  used a convolution-deconvolution procedure to precisely position transcription factor binding site (TFBS) within windows known to contain such a site.
The training in done in two steps. Firstly, the TFBS is positioned at the center of the sequence.
Secondly it is positioned arbitrarily in the window. To create the corresponding `Generator` instance, the position of the TFBS can be passed through a BED file (constructed after analysing the ChIP-seq data).
For the first training process, after choosing the training chromosomes and specifying the sequence length, one needs just to specify that the problem is a sequence to sequence problem (by setting the keyword `seq2seq` to True).
For the second training, a new `Generator` must be created following the previous procedure, and a data augmentation procedure can be applied, as the in the original study (set `data_augmentation` to True).

```python
from keras_dna import Generator, ModelWrapper

### Suppose we have a function to create and compile the model
model = create_model()

### First training
generator = Generator(batch_size=64,
                      fasta_file='hg19.fa',
                      annotation_file=['ChIPseq_peaks.bed'],
                      annotation_list=['binding site'],
                      seq_len=100,
                      seq2seq=True,
                      data_augmentation=False,
                      negative_type=None,
                      excl_chromosomes=['chr19', 'chr20', 'chr21'])

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr19'])
wrap.train(epochs=10)

### Second training
generator = Generator(batch_size=64,
                      fasta_file='hg19.fa',
                      annotation_file=['ChIPseq_peaks.bed'],
                      annotation_list=['binding site'],
                      seq_len=100,
                      seq2seq=True,
                      data_augmentation=True,
                      negative_type=None,
                      excl_chromosomes=['chr19', 'chr20', 'chr21'])

wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr19'])
wrap.train(epochs=10)
```

## Multi-regression

[Kelley et al.](https://genome.cshlp.org/content/28/5/739.short) designed a convolutional neural network to predict the regulatory activities from sequences.
The goal was to predict the read coverage coming from several ChIP-seq, CAGE or DNase-seq experiments in a 128-bp bins across chromosomes.
The data are available in BIGWIG format and can be handled with a `Generator` instance, passing those files as inputs. 
One needs to specify the window length to 2^17 (keyword `window`), the target coverage sequence length to 2^17 / 128 = 2^10 (keyword `tg_window`) and a binning procedure needs to be applied (keyword `downsampling`).
Once the generator and the model are associated in a `ModelWrapper` it is easy to predict on a whole chromosome and the predicted coverages can be exported one by one in the BIGWIG format.

```python
from keras_dna import Generator, ModelWrapper

### Suppose we have a function to create and compile the model
model = create_model()

generator = Generator(batch_size=64,
                      fasta_file='hg38.fa',
                      annotation_files=['artery_endotellial_CAGE.bw', 'artery_endotellial_ChIPseq.bw', 'artery_endotellial_DNase.bw',
                                        'heart_CAGE.bw', 'heart_ChIPseq.bw', 'heart_DNase.bw'],
                      nb_annotation_type=3,
                      window=2**17,
                      tg_window=2**10,
                      downsampling='downsampling',
                      excl_chromosomes=['chr19', 'chr20', 'chr21'])
                      
>>> generator.label_shape
(1024, 2, 3)
 
wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr19'])
wrap.train(epochs=10)

### Predict on specific loci to see the effect of variant
wrap.predict(incl_chromosomes=['chr20', 'chr21'],
             start_stop=[(BIRC3_start, BIRC3_stop), (FZD1_start, FZD1_stop)]
             chrom_size='hg38.chrom.sizes',
             export_to_path='wild_type_prediction.bw')
             
wrap.predict(incl_chromosomes=['chr20', 'chr21'],
             fasta_file='hg38_variant.fa',
             start_stop=[(BIRC3_start, BIRC3_stop), (FZD1_start, FZD1_stop)]
             chrom_size='hg38.chrom.sizes',
             export_to_path='wild_type_prediction.bw')

```

## Multi-input classification

[Nair et al.](https://academic.oup.com/bioinformatics/article/35/14/i108/5529138) used a deep learning methodology to predict the chromatin accessibility genome-wide across a cellular context using both the DNA sequence and the cellular type dependant RNA-seq as input.
The data pipeline can efficiently be created using keras_dna. A `Generator` instance can be created as seen previously, but one would also need to pass the files containing the RNA-seq coverage as secondary inputs (using the keyword `sec_inputs`).
`Generator`class owns the same functionalities for continuous secondary inputs as for continuous targets.

```python
from keras_dna import Generator, ModelWrapper

### Suppose we have a function to create and compile the model
model = create_model()

### Suppose the genes data are stucked in a gff file
generator = Generator(batch_size=64,
                      fasta_file='hg38.fa',
                      annotation_files=['binary_peaks_cell1.bed', ..., 'binary_peaks_cell123.bed'],
                      seq_len=1000,
                      annotation_list=['dna accessibility'],
                      negative_ratio=3,
                      sec_inputs=['rna_seq_cell1.bw', ..., 'rna_seq_cell123.bw'],
                      sec_input_shape=(64, 1000, 123),
                      sec_normalization_mode=['max', 'perctrim'],
                      excl_chromosomes=['chr19', 'chr20', 'chr21'],
                      output_shape=(64, 123))
                      
wrap = ModelWrapper(model=model,
                    generator_train=generator,
                    validation_chr=['chr19', 'chr20'])
                    
wrap.train(epochs=20)
```






