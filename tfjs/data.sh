#!/bin/bash

set -evx

baseball(){

  DIR=./tmp/data/baseball

  mkdir -p $DIR

  cd $DIR

  wget https://storage.googleapis.com/mlb-pitch-data/pitch_type_training_data.csv
  wget https://storage.googleapis.com/mlb-pitch-data/pitch_type_test_data.csv

}

"$@"