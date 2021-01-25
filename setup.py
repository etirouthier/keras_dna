import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE/"README.md").read_text()

setup(name="keras_dna",
      version="0.0.39",
      description="Build keras generator for genomic application",
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/etirouthier/keras_dna",
      author="Etienne Routhier",
      author_email="etienne.routhier@upmc.fr",
      license="MIT",
      classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6"],
      packages=find_packages(),
      include_packages_data=True,
      install_requires=["tensorflow>=2.0.0",
                        "numpy",
                        "pandas",
                        "kipoiseq",
                        "pyBigWig",
                        "pybedtools"]
      )
      
