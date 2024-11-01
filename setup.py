from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'torch',
    'torch-hd',
    'torchvision',
    'torchaudio',
    'matplotlib',
    'dvu',  # visualization
    'pandas',
    'transformers',
    'openai',
    'accelerate',
    'datasets',
    'infini-gram',
    'pydivsufsort',
    'nltk',
    'spacy',
    'ansicolors',
    'datalad',
    'h5py',
    'huggingface-hub',
    'imageio',
    'numpy',
    'safetensors',
    'scikit-learn',
    'scipy',
    'sentence-transformers',
    'sentencepiece',
    'simplejson',
    'six',
    'smmap',
    'sympy',
    'tables',
    'tabulate',
    'tdqm',
    'tokenizers',
    'tqdm',
    'bitsandbytes',
    'tiktoken'
]


setuptools.setup(
    name="alm and encoding",
    version="0.01",
    author="Microsoft Research",
    author_email="",
    description="Experiments with alternative approaches to language modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/gpt-alt-lm",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required_pypi,
)


