#!/bin/bash

bert_base(){
  set -evx

  DIR=./tmp/data/bert-base

  mkdir -p $DIR

  cd $DIR

  wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

  unzip *.zip

  mv ./uncased_L-12_H-768_A-12/* ./
}

"$@"