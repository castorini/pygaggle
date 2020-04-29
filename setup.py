import setuptools


with open('README.md') as fh:
    long_description = fh.read()

reqs = [
    'dataclasses;python_version<"3.7"',
    'coloredlogs==14.0',
    'numpy==1.18.2',
    'pydantic==1.5',
    'pyserini==0.9.0.0',
    'scikit-learn>=0.22',
    'scipy>=1.4',
    'spacy==2.2.4',
    'tensorboard>=2.1.0',
    'tensorflow>=2.2.0rc1',
    'tokenizers>=0.5.2',
    'tqdm==4.45.0',
    'transformers>=2.7.0'
]

setuptools.setup(
    name='pygaggle',
    version='0.0.1',
    author='PyGaggle Gaggle',
    author_email='r33tang@uwaterloo.ca',
    description='A gaggle of rerankers for CovidQA and CORD-19',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/castorini/pygaggle',
    install_requires=reqs,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)