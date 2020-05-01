#!/bin/bash

mkdir ../data/embeddings
cd ../data/embeddings

wget http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip


