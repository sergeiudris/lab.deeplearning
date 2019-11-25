FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
# supported tags
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md

## core
RUN apt-get update && \
    apt-get install -y \
            sudo  \
            git-core  \
            rlwrap  \
            software-properties-common  \
            unzip wget curl net-tools lsof \
            graphviz \
            zlib1g-dev gcc libc6-dev

WORKDIR /tmp

## openjdk
# RUN apt-get update && \
#     apt-get install -y openjdk-8-jdk

## graalvm
# https://github.com/arjones/docker-graalvm/blob/master/Dockerfile
# https://github.com/OlegIlyenko/graalvm-native-image/blob/master/Dockerfile
ENV GRAALVM_VERSION=19.3.0
ENV SUFFIX_URL=java8-linux-amd64-${GRAALVM_VERSION}
ENV SUFFIX_DIR=java8-${GRAALVM_VERSION}
ENV PATH $PATH:/usr/local/graalvm/bin
#  dir will be graalvm-ce-java8-19.3.0
RUN curl -Ls "https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-19.3.0/graalvm-ce-${SUFFIX_URL}.tar.gz" | \
    tar zx -C /usr/local/ && \
    ls -l /usr/local/ && \
    rm -f /usr/local/graalvm-ce-${SUFFIX_DIR}/src.zip && \
    ln -s /usr/local/graalvm-ce-${SUFFIX_DIR} /usr/local/graalvm && \
    rm -fr /var/lib/apt
RUN gu install native-image 

## clojure
ENV CLOJURE_TOOLS=linux-install-1.10.1.466.sh
RUN curl -O https://download.clojure.org/install/$CLOJURE_TOOLS && \
    chmod +x $CLOJURE_TOOLS && \
    sudo ./$CLOJURE_TOOLS && \
    clojure -Stree

## leiningen
ENV LEIN_VERSION=2.9.1
ENV LEIN_DIR=/usr/local/bin/
RUN curl -O https://raw.githubusercontent.com/technomancy/leiningen/${LEIN_VERSION}/bin/lein && \
    mv lein ${LEIN_DIR} && \
    chmod a+x ${LEIN_DIR}/lein && \
    lein version

## node
RUN curl -sL https://deb.nodesource.com/setup_11.x | sudo -E bash - && \
    sudo apt-get install -y nodejs 
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list && \
    sudo apt-get update && sudo apt-get -y install yarn

# ## opencv
# RUN sudo add-apt-repository ppa:timsc/opencv-3.4 && \
#     apt-get update && \
#     apt-get install -y  \
#         libopencv-imgcodecs3.4 \
#         libopenblas-base \
#         libatlas3-base \
#         libcurl3 

RUN apt-get update && \
        apt-get install -y curl

# /usr/local/graalvm-ce-java8-19.3.0/jre/lib/amd64/libawt_xawt.so: libXtst.so.6: cannot open shared object file: No such file or directory
# /usr/local/graalvm-ce-java8-19.3.0/jre/lib/amd64/libawt_xawt.so: libXi.so.6: cannot open shared object file: No such file or directory
RUN apt-get install -y libxtst6 libxi6

## app
WORKDIR /opt/app
COPY project.clj .
RUN lein deps

EXPOSE 7788 8080
CMD ["bash","c", "dev"]