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

link_spaces() {
    SPACE=samples
    mkdir -p spaces/$SPACE

    ln -s ../../.vscode spaces/$SPACE/.vscode
    ln -s ../../samples/app/deps.edn spaces/$SPACE/deps.edn


    mkdir -p spaces/$SPACE/tln

    ln -s ../../../samples/tln/src/app/tln spaces/$SPACE/tln/src
    ln -s ../../../samples/tln/.data spaces/$SPACE/tln/.data

}

"$@"