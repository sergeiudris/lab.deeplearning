#!/bin/bash

set -evx

fashion_mnist(){

  DIR=./tmp/data/fashion-mnist
  mkdir -p $DIR
  cd $DIR

  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

  gunzip *.gz

}

mnist(){

  DIR=./tmp/data/mnist
  mkdir -p $DIR
  cd $DIR

  wget http://data.mxnet.io/mxnet/data/mnist.zip 
  unzip -u mnist.zip
}

house(){
  DIR=./tmp/data/house
  mkdir -p $DIR
  cd $DIR

  # wget https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv
  # wget https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/test.csv

  wget https://s3.us-east-2.amazonaws.com/tech.public.data/house-prices-advanced-regression-techniques.zip
  unzip house-prices-advanced-regression-techniques.zip
}

ner(){

  # https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
  
  # cannot download via command line, so use the link
  https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/downloads/ner_dataset.csv

  # an example of a mv command
  sudo mv *.csv /home/user/code/sample.ml/d2l/tmp/data/ner/
}

viz(){

  DIR=./tmp/data/viz
  mkdir -p $DIR
  cd $DIR

  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params

  wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json
  wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params

}

inception(){

  DIR=./tmp/data/inception
  mkdir -p $DIR
  cd $DIR

  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params

  wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-symbol.json
  wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-0126.params
  mv Inception-BN-0126.params Inception-BN-0000.params

  wget http://data.mxnet.io/models/imagenet/synset.txt

  # images
  wget https://arthurcaillau.com/assets/images/cat-egyptian.jpg
  wget https://arthurcaillau.com/assets/images/dog-2.jpg
  wget https://arthurcaillau.com/assets/images/guitarplayer.jpg
}

glove(){

  DIR=./tmp/data/glove
  mkdir -p $DIR
  cd $DIR

  wget http://nlp.stanford.edu/data/glove.6B.zip
  
  unzip *.zip

}

bert(){

  DIR=./tmp/data/bert
  mkdir -p $DIR
  cd $DIR

  curl https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA/vocab.json -o vocab.json
  curl https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA/static_bert_qa-0002.params -o static_bert_qa-0002.params
  curl https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA/static_bert_qa-symbol.json -o static_bert_qa-symbol.json
  curl https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA/static_bert_base_net-symbol.json -o static_bert_base_net-symbol.json
  curl https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/BertQA/static_bert_base_net-0000.params -o static_bert_base_net-0000.params
  curl https://raw.githubusercontent.com/dmlc/gluon-nlp/master/docs/examples/sentence_embedding/dev.tsv -o dev.tsv

}

recom(){

  DIR=./tmp/data/recom

  mkdir -p $DIR
  cd $DIR

  wget http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
  tar -xvzf MovieSummaries.tar.gz
  mv ./MovieSummaries/* ./

}

bert_base(){

  MXNET_MODELS_DIR=/root/.mxnet/models
  MODELS_DIR="$(pwd)"/tmp/models

  mkdir -p $MODELS_DIR

  python3 python/bert.py
  mv $MXNET_MODELS_DIR/* $MODELS_DIR

}

export_bert(){
  # regression classification question_answering
  python3 python/bert/export/export.py \
    --task classification \
    --prefix "bert-cls-4" \
    --seq_length 512 \
    --num_classes 4 \
    --output_dir /opt/app/tmp/data/bert-base
}

fasttext(){
  DIR=./tmp/data/fasttext

  mkdir -p $DIR
  cd $DIR
  https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec
  
}

text8(){
  DIR=./tmp/data/text8

  mkdir -p $DIR
  cd $DIR
  wget https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/large_text_compression_benchmark/text8-6c70299b.zip
  unzip *.zip

}

"$@"