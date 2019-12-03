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

}

link_spaces_el(){
    SPACE=el
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../el/deps.edn spaces/$SPACE/deps.edn
    ln -s ../../el/src/el spaces/$SPACE/el
    ln -s ../../el/tmp spaces/$SPACE/tmp

    ln -s ../../spaces/mxnet spaces/$SPACE/mxnet
    ln -s ../../d2l/src/d2l spaces/$SPACE/d2l
    

}

link_spaces_dl4j(){
    SPACE=dl4j
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../dl4j/project.clj spaces/$SPACE/project.clj
    ln -s ../../dl4j/src/dl4j spaces/$SPACE/dl4j
    ln -s ../../dl4j/src/org spaces/$SPACE/org

    ln -s ../../dl4j/tmp spaces/$SPACE/tmp
}

link_spaces_lucene(){
    SPACE=lucene
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../lucene/project.clj spaces/$SPACE/project.clj
    ln -s ../../lucene/src/lucene spaces/$SPACE/lucene
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


}

link_spaces_tfjs(){
    SPACE=tfjs
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode

    ln -s ../../tfjs/shadow-cljs.edn spaces/$SPACE/shadow-cljs.edn
    ln -s ../../tfjs/src/app spaces/$SPACE/app
    ln -s ../../tfjs/tmp spaces/$SPACE/tmp
}

dock_u18(){
    docker run  --rm \
                --name ubuntu18 \
                --memory 16g \
                --cpus 4.000 \
                -it \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)":/opt/code/ \
                 ubuntu:18.04 \
                 bash -c "cd /opt/root/;bash"
}

dock_mmdnn(){
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
