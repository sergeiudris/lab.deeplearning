#!/bin/bash

link_spaces_samples() {
    SPACE=samples
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../samples/app/deps.edn spaces/$SPACE/deps.edn


    mkdir -p spaces/$SPACE/tln

    ln -s ../../../samples/tln/src/app/tln spaces/$SPACE/tln/src
    ln -s ../../../samples/tln/.data spaces/$SPACE/tln/.data

}

link_spaces_mxnet() {
    SPACE=mxnet
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode


    EXAMPLE=bert
    mkdir -p spaces/$SPACE/$EXAMPLE
    # ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
    ln -s ../../../mxnet/examples/$EXAMPLE/src spaces/$SPACE/$EXAMPLE/src
    ln -s ../../../mxnet/examples/$EXAMPLE/data spaces/$SPACE/$EXAMPLE/data

    EXAMPLE=rnn
    mkdir -p spaces/$SPACE/$EXAMPLE
    ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
    ln -s ../../../mxnet/examples/$EXAMPLE/src spaces/$SPACE/$EXAMPLE/src
    ln -s ../../../mxnet/examples/$EXAMPLE/data spaces/$SPACE/$EXAMPLE/data

    EXAMPLE=cnn-text-classification
    mkdir -p spaces/$SPACE/$EXAMPLE
    ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
    ln -s ../../../mxnet/examples/$EXAMPLE/src spaces/$SPACE/$EXAMPLE/src
    ln -s ../../../mxnet/examples/$EXAMPLE/data spaces/$SPACE/$EXAMPLE/data

    EXAMPLE=module
    mkdir -p spaces/$SPACE/$EXAMPLE
    ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
    ln -s ../../../mxnet/examples/$EXAMPLE/src spaces/$SPACE/$EXAMPLE/src
    ln -s ../../../mxnet/examples/$EXAMPLE/data spaces/$SPACE/$EXAMPLE/data

    EXAMPLE=tutorial
    mkdir -p spaces/$SPACE/$EXAMPLE
    ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
    ln -s ../../../mxnet/examples/$EXAMPLE/src spaces/$SPACE/$EXAMPLE/src
    ln -s ../../../mxnet/examples/$EXAMPLE/data spaces/$SPACE/$EXAMPLE/data

    EXAMPLE=imclassification
    mkdir -p spaces/$SPACE/$EXAMPLE
    ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
    ln -s ../../../mxnet/examples/$EXAMPLE/src spaces/$SPACE/$EXAMPLE/src
    ln -s ../../../mxnet/examples/$EXAMPLE/data spaces/$SPACE/$EXAMPLE/data

    EXAMPLE=pre-trained-models
    mkdir -p spaces/$SPACE/$EXAMPLE
    ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
    ln -s ../../../mxnet/examples/$EXAMPLE/src spaces/$SPACE/$EXAMPLE/src
    ln -s ../../../mxnet/examples/$EXAMPLE/model spaces/$SPACE/$EXAMPLE/model

    EXAMPLE=captcha
    mkdir -p spaces/$SPACE/$EXAMPLE
    ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
    ln -s ../../../mxnet/examples/$EXAMPLE/src spaces/$SPACE/$EXAMPLE/src
    ln -s ../../../mxnet/examples/$EXAMPLE/data spaces/$SPACE/$EXAMPLE/data

}

link_spaces_d2l(){
    SPACE=d2l
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode

    ln -s ../../d2l/deps.edn spaces/$SPACE/deps.edn
    ln -s ../../d2l/src/d2l spaces/$SPACE/d2l
    ln -s ../../spaces/mxnet spaces/$SPACE/mxnet
    ln -s ../../d2l/tmp spaces/$SPACE/tmp
    ln -s ../../../pad/spaces/pad spaces/$SPACE/pad


}

link_spaces_el(){
    SPACE=el
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../el/deps.edn spaces/$SPACE/deps.edn
    ln -s ../../el/src/el spaces/$SPACE/el
    ln -s ../../el/tmp spaces/$SPACE/tmp

    ln -s ../../../pad/spaces/pad spaces/$SPACE/pad
    ln -s ../../spaces/mxnet spaces/$SPACE/mxnet
    ln -s ../../d2l/src/d2l spaces/$SPACE/d2l
    

}

link_spaces_gvm(){
    SPACE=gvm
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../gvm/deps.edn spaces/$SPACE/deps.edn
    ln -s ../../gvm/src/gvm spaces/$SPACE/gvm
    ln -s ../../gvm/tmp spaces/$SPACE/tmp

    ln -s ../../../pad/spaces/pad spaces/$SPACE/pad
    ln -s ../../spaces/mxnet spaces/$SPACE/mxnet
    ln -s ../../d2l/src/d2l spaces/$SPACE/d2l
    

}

link_spaces_dl4j(){
    SPACE=dl4j
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../dl4j/project.clj spaces/$SPACE/project.clj
    ln -s ../../dl4j/src/dl4j spaces/$SPACE/dl4j
    ln -s ../../dl4j/src/pad spaces/$SPACE/pad
    ln -s ../../dl4j/src/org spaces/$SPACE/org

    ln -s ../../dl4j/tmp spaces/$SPACE/tmp
}

link_spaces_lucene(){
    SPACE=lucene
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../lucene/project.clj spaces/$SPACE/project.clj
    ln -s ../../lucene/src/lucene spaces/$SPACE/lucene
    ln -s ../../lucene/src/pad spaces/$SPACE/pad
    ln -s ../../lucene/src/org spaces/$SPACE/org

    ln -s ../../lucene/tmp spaces/$SPACE/tmp
}

link_spaces_tf(){
    SPACE=tf
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode

    ln -s ../../tf/deps.edn spaces/$SPACE/deps.edn
    ln -s ../../tf/src/tf spaces/$SPACE/tf
    ln -s ../../tf/tmp spaces/$SPACE/tmp
    ln -s ../../../pad/spaces/pad spaces/$SPACE/pad


}

link_spaces_tfjs(){
    SPACE=tfjs
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode

    ln -s ../../tfjs/shadow-cljs.edn spaces/$SPACE/shadow-cljs.edn
    ln -s ../../tfjs/src/app spaces/$SPACE/app
    ln -s ../../tfjs/tmp spaces/$SPACE/tmp
    ln -s ../../../pad/spaces/pad spaces/$SPACE/pad
}

permissions(){
    sudo chmod -R 777 d2l/tmp/ 
}


mxnet() {
    docker build -t sample.ml.mxnet mxnet/
    docker run --gpus all \
                --rm \
                --name mxnet \
                -it \
                -p 7788:7888 \
                -v "$(pwd)"/mxnet:/opt/app \
                -v "$(pwd)":/opt/root \
                -v "$(pwd)"/mxnet/.data:/opt/data \
                 sample.ml.mxnet \
                 bash
}

d2l() {
    docker build -t sample.ml.d2l d2l/
    docker run --gpus all \
                --rm \
                --name d2l \
                -it \
                -p 7788:7888 \
                -v "$(pwd)"/d2l:/opt/app \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)"/pad:/opt/code/pad \
                 sample.ml.d2l \
                 bash
}

tf() {
    docker build -t sample.ml.tf tf/
    docker run --gpus all \
                --rm \
                --name tf \
                -it \
                -p 7788:7888 \
                -v "$(pwd)"/tf:/opt/app \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)"/pad:/opt/code/pad \
                 sample.ml.tf \
                 bash
}

tfjs() {
    docker build -t sample.ml.tfjs tfjs/
    docker run --gpus all \
                --rm \
                --name tfjs \
                -it \
                -p 7788:7888 \
                -v "$(pwd)"/tfjs:/opt/app \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)"/pad:/opt/code/pad \
                 sample.ml.tfjs \
                 bash
}

gvm() {
    docker build -t sample.ml.gvm gvm/
    docker run --gpus all \
                --rm \
                --name gvm \
                --memory 16g \
                --cpus 4.000 \
                -it \
                -p 7788:7888 \
                -v "$(pwd)"/gvm:/opt/app \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)"/pad:/opt/code/pad \
                 sample.ml.gvm \
                 bash
}

gvm_samples() {
    docker build -t sample.ml.gvm-samples gvm-samples/
    docker run  --rm \
                --name gvm-samples \
                --memory 16g \
                --cpus 4.000 \
                -it \
                -p 7788:7888 \
                -v "$(pwd)"/gvm-samples:/opt/app \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)"/pad:/opt/code/pad \
                 sample.ml.gvm-samples \
                 bash
}


dl4j() {
    sudo xhost +
    docker build -t sample.ml.dl4j dl4j/
    docker run  --gpus all \
                --env DISPLAY=${DISPLAY} \
                -v /tmp/.X11-unix:/tmp/.X11-unix \
                --rm \
                --name dl4j \
                --memory 12g \
                --cpus 4.000 \
                -it \
                -p 7788:7788 \
                -p 9000:9000 \
                -v "$(pwd)"/dl4j:/opt/app \
                -v "$(pwd)"/dl4j/.deeplearning4j:/root/.deeplearning4j \
                -v "$(pwd)"/dl4j/dl4j-examples-data:/root/dl4j-examples-data \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)"/pad:/opt/code/pad \
                 sample.ml.dl4j \
                 bash
}

lucene() {
    sudo xhost +
    docker build -t sample.ml.lucene lucene/
    docker run  --gpus all \
                --env DISPLAY=${DISPLAY} \
                -v /tmp/.X11-unix:/tmp/.X11-unix \
                --rm \
                --name lucene \
                --memory 12g \
                --cpus 4.000 \
                -it \
                -p 7788:7788 \
                -v "$(pwd)"/lucene:/opt/app \
                -v "$(pwd)":/opt/root \
                 sample.ml.lucene \
                 bash
}

u18(){
    docker run  --rm \
                --name ubuntu18 \
                --memory 16g \
                --cpus 4.000 \
                -it \
                -p 8888:7888 \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)":/opt/code/ \
                 ubuntu:18.04 \
                 bash -c "cd /opt/root/;bash"
}

mmdnn(){
    docker run \
            --rm \
            --name mmdnn \
            -it \
            -p 6006:6006 \
            -v "$(pwd)":/opt/root \
                mmdnn/mmdnn:cpu.small \
                bash -c "cd /opt/root;bash;"
}

convert_bert(){
    DIR_BERT=/opt/root/d2l/tmp/data/recom/uncased_L-12_H-768_A-12
    DIR=/opt/root/d2l/tmp/data/recom/converted
    mkdir -p $DIR
    cd $DIR
    mmconvert \
            -sf tensorflow \
            -in  $DIR_BERT/bert_model.ckpt.meta \
            -iw $DIR_BERT/bert_model.ckpt \
            --dstNodeName "should be output node of the model" \
            -df mxnet \
            -om bert_model.mxnet
}

clean_converted(){
    DIR=/opt/root/d2l/tmp/data/recom/converted
    rm -rf $DIR/*
}

viz_bert(){
    cd d2l/tmp/data/recom/uncased_L-12_H-768_A-12
    mmvismeta bert_model.ckpt.meta ./logs/ 
}

tensorboard(){
    tensorboard --logdir ./logs/ --port 6006 --host "0.0.0.0"
}


"$@"
