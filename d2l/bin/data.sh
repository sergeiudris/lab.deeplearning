#!/bin/bash



fashion_mnist(){

  set -evx

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

  set -evx

  DIR=./tmp/data/mnist

  mkdir -p $DIR

  cd $DIR

  wget http://data.mxnet.io/mxnet/data/mnist.zip 
  unzip -u mnist.zip
}

house(){
   set -evx

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
  set -evx

  DIR=./tmp/data/viz

  mkdir -p $DIR

  cd $DIR
  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params

  wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json
  wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params

}

inception(){

  set -evx

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

"$@"