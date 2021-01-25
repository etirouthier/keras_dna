from __future__ import absolute_import

from . import evaluation
from . import normalization
from . import extractors
from . import generators
from . import layers
from . import model
from . import utils
from . import sequence


# Also importable from root
from .generators import Generator, MultiGenerator
from .model import ModelWrapper
from .sequence import SeqIntervalDl, StringSeqIntervalDl

__version__ = '0.0.38'


def get_ucsc():
    import os, stat
    distrib = ''
    while distrib not in {"linux.x86_64.v369", 
                           "linux.x86_64.v385",
                           "linux.x86_64",
                           "macOSX.x86_64"}:
        distrib = input("Choose your OS: (linux.x86_64.v369," + \
        "linux.x86_64.v385, linux.x86_64, macOSX.x86_64)\n")
    
    path = input("Enter the path to the directory where you want to save the UCSC utils.\n")

    if not os.path.exists(path):
        os.mkdir(path)

    os.system("wget -O " + path + '/wigToBigWig https://hgdownload.soe.ucsc.edu/admin/exe/' + distrib + '/wigToBigWig')
    os.system("wget -O " + path + '/bedGraphToBigWig https://hgdownload.soe.ucsc.edu/admin/exe/' + distrib + '/bedGraphToBigWig')

    os.chmod(path + '/wigToBigWig', stat.S_IEXEC)
    os.chmod(path + '/bedGraphToBigWig', stat.S_IEXEC)
