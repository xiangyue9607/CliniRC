# Clinical Reading Comprehension (CliniRC)

## Introduction
This repository provides code for the analysis of Clinical Reading Comprehension task in the ACL2020 paper:
[Clinical Reading Comprehension: A Thorough Analysis of the emrQA Dataset](https://arxiv.org/abs/2005.00574)

```bib
@inproceedings{yue2020CliniRC,
 title={Clinical Reading Comprehension: A Thorough Analysis of the emrQA Dataset},
 author={Xiang Yue and Bernal Jimenez Gutierrez and Huan Sun},
 booktitle={ACL},
 year={2020}
}
```

## Set up
Run the following commands to clone the repository and install requirements. 
It requires Python 3.5 or higher. 
It also requires installing [PyTorch](https://pytorch.org/) version 1.0 or higher and [Tensorflow](https://www.tensorflow.org/) version 1.1 or higher.
The other dependencies are listed in requirements.txt. 
```shell script
$ git clone https://github.com/xiangyue9607/CliniRC.git
$ pip install -r requirements.txt 
```

## Preparing the emrQA Dataset
Our analysis is based on the recently released clinical QA dataset: emrQA [[EMNLP'18]](https://arxiv.org/abs/1809.00732). 
Note that the emrQA dataset is generated from the [n2c2 (previously "i2b2") datasets](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). 
**We do not have the right to include either the emrQA dataset or n2c2 datasets in this repo.** 
Users need to first sign up the n2c2 data use agreement and then follow the instructions in the [emrQA repo](https://github.com/panushri25/emrQA) to generate the emrQA dataset. 
After you generate the emrQA dataset, create directory ``/data/datasets`` and put the ``data.json`` into the directory.

## Preprocessing
We first provide proprocessing script to help clean up the generated emrQA dataset. 
Specifically, the preprocessing script have the following functions:
1) Remove extra whitespaces, punctuations and newlines. Join sentences into one paragraph;
2) Reformulate the dataset as the "[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)" format;
3) Randomly split the dataset into train/dev/test set (7:1:2).

```shell script
$ python src/preprocessing.py \
--data_dir ./data/datasets \
--filename data.json \
--out_dir ./data/datasets
```

Note that there are 5 subsets in the emrQA dataset. We only use the ``Medication`` and ``Relation`` subsets, as (1) they makeup 80% of the entire emrQA dataset and 
(2) their format is consistent with the span extraction task, which is more challenging and meaningful for clinical decision making support. 

After running ``preprocessing.py`` script, you will obtain 6 json files in your ``output`` directory (i.e., train, dev, test sets for ``Medication`` and ``Relation`` datasets) 

## Sampling Subsets to accelerate training
As we have demonstrated in the paper (Section 4.1), though there are more than 1 million questions in the emrQA dataset, 
many questions and their patterns are very similar since they are generated from the same question templates. 
And we show that we do not so many questions to train a CliniRC system and using a sampled subset can achieve roughly the same performance that is based on the entire dataset.

To randomly sample question from the original dataset, you can:

```shell script
$ python src/sample_dataset.py \
--data_dir ./data/datasets \
--filename medication-train \
--out_dir ./data/datasets \
--sample_ratio 0.2
```

```shell script
$ python src/sample_dataset.py \
--data_dir ./data/datasets \
--filename relation-train \
--out_dir ./data/datasets \
--sample_ratio 0.05
```

--sample_ratio controls how many questions are sampled from each document.

## Train and Test a QA model
In our paper, we compare some state-of-tha-art QA models on the emrQA dataset. Here, we give two examples: [BERT](https://arxiv.org/abs/1810.04805) and [DocReader](https://arxiv.org/abs/1704.00051). 
For other QA models tested in the paper, you can refer to their github repos for further details.
### BERT
1. Download the pretrained BERT models (including [bert-base-cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip), 
    [BioBERT-base-PubMed](https://drive.google.com/file/d/1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD/view?usp=sharing) and 
    [ClinicalBERT](https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=0)). (Feel free to try other BERT models-:)
```shell script
$ chmod +x download_pretrained_models.sh; ./download_pretrained_models.sh
```
2. Train (Fine-tune) a BERT model on the emrQA medication/relation dataset. The training script is adopted from [BERT github repo](https://github.com/google-research/bert)
```shell script
$ CUDA_VISIBLE_DEVICES=0 python ./BERT/run_squad.py \
    --vocab_file=./pretrained_bert_models/clinicalbert/vocab.txt \
    --bert_config_file=./pretrained_bert_models/clinicalbert/bert_config.json \
    --init_checkpoint=./pretrained_bert_models/clinicalbert/model.ckpt-100000 \
    --do_train=True \
    --train_file=./data/datasets/relation-train-sampled-0.05.json \
    --do_predict=True \
    --do_lower_case=False \
    --predict_file=./data/datasets/relation-dev.json \
    --train_batch_size=6 \
    --learning_rate=3e-5 \
    --num_train_epochs=4.0 \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=./output/bert_models/clinicalbert_relation_0.05/
```
3. Inference on the test set.
```shell script
$ python ./BERT/run_squad.py \
    --vocab_file=./pretrained_bert_models/clinicalbert/vocab.txt \
    --bert_config_file=./pretrained_bert_models/clinicalbert/bert_config.json \
    --init_checkpoint=./output/bert_models/clinical_relation_0.05_epoch51/model.ckpt-21878 \
    --do_train=False \
    --do_predict=True \
    --do_lower_case=False \
    --predict_file=./data/relation-test.json \
    --train_batch_size=6 \
    --learning_rate=3e-5 \
    --num_train_epochs=3.0 \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=./output/bert_models/clinical_relation_0.05_epoch51_test/
```
4. Eval the model. We adopt the official eval script from SQuAD v1.1. 
```shell script
$ python ./src/evaluate-v1.1.py ./data/datasets/medication-dev.json ./output/bert_models/bertbase_medication_0.2/predictions.json
```
### DocReader
We adopt the DocReader module code from [DrQA github repo](https://github.com/facebookresearch/DrQA).
1. Set up
```shell script
$ git clone https://github.com/facebookresearch/DrQA.git
$ cd DrQA; python setup.py develop
```
2. Download the pretrained [GloVe embeddings](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and put it into the ``data/embeddings``. You can also run our script to automatically finish this step:
```shell script
$ chmod +x ../download_glove_embeddings.sh; ../download_glove_embeddings.sh
```
3. Preprocessing the train/dev files:
```shell script
$ python scripts/reader/preprocess.py \
../data/datasets/ \
../data/datasets/ \
--split relation-train-sampled-0.05 \
--tokenizer spacy
```

```shell script
$ python scripts/reader/preprocess.py \
../data/datasets/ \
../data/datasets/ \
--split relation-dev \
--tokenizer spacy
```
4. Train the Reader:
```shell script
$ python scripts/reader/train.py \
--embedding-file glove.840B.300d.txt \
--tune-partial 1000 \
--train-file relation-train-sampled-0.05-processed-spacy.txt \
--dev-file relation-dev-processed-spacy.txt \
--dev-json relation-dev.json \
--random-seed 20 \
--batch-size 16 \
--test-batch-size 16 \
--official-eval True \
--valid-metric exact_match \
--checkpoint True \
--model-dir ../output/drqa-models/relation \
--data-dir ../data/datasets \
--embed-dir ../data/embeddings \
--data-workers 0 \
--max-len 30 
```
5. Inference on the test set:
```shell script
python scripts/reader/predict.py \
../data/datasets/relations-mimic-new-qs-ver3.json \
--model ../output/drqa-models/[YOUR MODEL NAME] \
--batch-size 16 \
--official \
--tokenizer spacy \
--out-dir ../output/drqa-models/ \
--embedding ../data/embeddings/glove.840B.300d.txt \
```
6. Eval the model. We adopt the official eval script from SQuAD v1.1. 
```shell script
$ cd ..
$ python ./src/evaluate-v1.1.py ./data/datasets/medication-dev.json ./output/drqa-models/predictions.json
```
