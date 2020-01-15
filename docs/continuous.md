# Generating continuous data

## Introduction

`Generator` can be use to feed a keras model with data that are annotated with a continuous function such as MNase or Chip-seq. The inputs file will then be a bigWig file, it can also be a wig or a bedGraph file but then another file containing the length of chromosome is needed (a conversion will be made using Kent's tools, two columns chrom, size tab separated). `Generator` will detect such files with the suffix .bw, .wig, .bedGraph. The length of the generated sequence needs to be passed with the keyword `window`.

```python
from keras_dna import Generator

generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.bw'],
                      window=299)

### With a wig file
generator = Generator(batch_size=64,
                      fasta_file='species.fa',
                      annotation_files=['ann.wig'],
                      size='species.chrom.sizes',
                      window=299)
```

The standard behaviour will be to generate all the sequence of window length in the DNA (one-hot-encoded) and to label it with the coverage at the centre nucleotide. The standard shape of DNA sequence is `(batch_size, window, 4)` and the standard labels shape is `(batch_size, tg_window, len(annotation_files))`, here `(64, 1, 1)`

## 
