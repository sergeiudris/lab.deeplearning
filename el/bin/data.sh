#!/bin/bash

set -evx

export_bert(){
  # regression classification question_answering
  python3 /opt/root/python/bert/export/export.py \
    --task classification \
    --prefix "bert-cls-4" \
    --seq_length 512 \
    --num_classes 4 \
    --output_dir /opt/app/tmp/data/bert-base
}

wiki_sample(){

  # https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20191101/

  DIR=./tmp/data/wiki-sample
  mkdir -p $DIR
  cd $DIR

  FILE=enwiki-20191101-pages-articles1.xml-p10p30302.bz2

  wget https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20191101/$FILE
  bzip2 -d $FILE

}

wiki(){

   # https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20191101/

  DIR=./tmp/data/wiki
  mkdir -p $DIR
  cd $DIR

  FILE=enwiki-20191101-pages-articles.xml.bz2

  wget https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/20191101/$FILE
  bzip2 -d $FILE

}


"$@"