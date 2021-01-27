import setuptools


with open('README.md') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='pygaggle',
    version='0.0.2',
    author='PyGaggle Gaggle',
    author_email='rpradeep@uwaterloo.ca',
    description='A gaggle of rerankers for text ranking and question answering.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/castorini/pygaggle',
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
