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
    ln -s ../../../mxnet/examples/$EXAMPLE/project.clj spaces/$SPACE/$EXAMPLE/project.clj
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

}


"$@"