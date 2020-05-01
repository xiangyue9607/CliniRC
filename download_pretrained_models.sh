#!/bin/bash

mkdir ./pretrained_bert_models
cd pretrained_bert_models

##Download BERT-base-cased
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip -O cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip
rm cased_L-12_H-768_A-12.zip

##Download BioBERT-base-PubMed_v1.1
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD" -O biobert_v1.1_pubmed.tar.gz  && rm -rf /tmp/cookies.txt
tar -xvf biobert_v1.1_pubmed.tar.gz
rm biobert_v1.1_pubmed.tar.gz

##Download ClinicalBERT
wget https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1 -O clinicalbert.tar.gz
tar -xvf clinicalbert.tar.gz
tar -xvf ./pretrained_bert_tf/biobert_pretrain_output_disch_100000.tar.gz
mv ./biobert_pretrain_output_disch_100000 ./clinicalbert
rm -rf ./pretrained_bert_tf/
rm clinicalbert.tar.gz


