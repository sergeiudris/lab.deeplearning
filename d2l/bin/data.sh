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

"$@"