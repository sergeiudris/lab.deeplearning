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
    SPACE=srv
    mkdir -p spaces/$SPACE
    # ln -s ../../app/src spaces/$SPACE/app
    # ln -s ../../.vscode spaces/$SPACE/.vscode
    # ln -s ../../app/deps.edn spaces/$SPACE/deps.edn
    # ln -s ../../app/.data spaces/$SPACE/.data

    mkdir -p spaces/$SPACE/examples
    ln -s ../../../examples/translation/src/examples/translation spaces/$SPACE/examples/translation

}

"$@"