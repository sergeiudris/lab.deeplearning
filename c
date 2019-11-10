#!/bin/bash

dc(){

    docker-compose --compatibility \
        -f docker-compose.yml \
        "$@"
}

up(){
    dc up -d --build
}

down(){
    dc down 
}

term(){
   dc exec $1 bash -c "bash;"
}

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

link_spaces_tf(){
    SPACE=tf
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode

    ln -s ../../tf/deps.edn spaces/$SPACE/deps.edn
    ln -s ../../tf/src/tf spaces/$SPACE/tf
    ln -s ../../tf/tmp spaces/$SPACE/tmp
    ln -s ../../../pad/spaces/pad spaces/$SPACE/pad


}

permissions(){
    sudo chmod -R 777 d2l/tmp/ 
}

d2l() {
    # use docker directly while docker-compose does not support --gpus flag
    # https://github.com/docker/compose/issues/6691
  
    docker run --gpus all \
                --rm \
                --name d2l \
                -it \
                -p 7888:7888 \
                -v "$(pwd)"/d2l:/opt/app \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)"/pad:/opt/code/pad \
                 sample.ml.d2l \
                 bash
}

tf() {
    # use docker directly while docker-compose does not support --gpus flag
    # https://github.com/docker/compose/issues/6691
  
    docker run --gpus all \
                --rm \
                --name tf \
                -it \
                -p 7878:7888 \
                -v "$(pwd)"/tf:/opt/app \
                -v "$(pwd)":/opt/root \
                -v "$(cd ../ && pwd)"/pad:/opt/code/pad \
                 sample.ml.tf \
                 bash
}

"$@"
