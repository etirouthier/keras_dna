# Guide to generators

-------------------------

Generators are the core part of keras_dna anabling to feed easily a keras model with the data corresponding to a given task with adapted format. 

--------------------------

### `Generator` class

- **Generating sparse data**

`Generator` can be used to feed a keras model with the subsequence of DNA underlying all the exemple of an annotation in the genome. To do so the position of annotations need to be passed in a bed or gff file, with the name of the target annotation in a list. `Generator` will detect suffixes .bed, .gff, .gtf, .gff3 and select the positions of the desired annotation.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['annotation.gff'],
                      annotation_list=['binding site'])
```

The standard behaviour will be to generate all the DNA sequences underlying the desired annotation one-hot-encoded and with a length corresponding to the maximal length encountered. It will also generate the same amounts of DNA sequences away from annotation regions. 

The one-hot-encoded DNA batch will have the shape `(batch_size, max length, 4)`. The labels will be 0 or 1 with the shape `(batch_size, nb anotation files, len(annotation_list))`, i.e here `(64, 1, 1)`.
