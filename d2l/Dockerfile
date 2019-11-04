# FROM ubuntu:18.04
FROM nvidia/cuda:9.2-devel-ubuntu18.04
# FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
# FROM nvidia/cuda:9.0-devel-ubuntu16.04
# FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# FROM nvidia/cuda:9.1-devel-ubuntu16.04
# FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# supported tags
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md

RUN apt-get update && \
    apt-get install -y \
            sudo  \
            git-core  \
        #     gpg  \
            rlwrap  \
            software-properties-common  \
            unzip wget curl net-tools lsof \
            graphviz

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk

ENV CLOJURE_SCRIPT_VERSION=linux-install-1.10.1.466.sh

RUN curl -O https://download.clojure.org/install/$CLOJURE_SCRIPT_VERSION && \
    chmod +x $CLOJURE_SCRIPT_VERSION && \
    sudo ./$CLOJURE_SCRIPT_VERSION

WORKDIR /opt

RUN sudo add-apt-repository ppa:timsc/opencv-3.4 && \
    apt-get update && \
    apt-get install -y  \
        libopencv-imgcodecs3.4 \
        libopenblas-base \
        libatlas3-base \
        libcurl3 

RUN apt-get install -y curl

# installing mxnet from source

# RUN apt-get install -y maven

# RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet && \
#         cd mxnet && \
#         git checkout v1.6.x && \
#         git submodule update --init --recursive

# RUN apt-get install -y cmake ninja-build libopenblas-dev

# RUN cd mxnet && \
#         mkdir build && cd build && \
#         cmake -DUSE_OPENCV=0 -DUSE_CUDA=1 -DUSE_CUDA_PATH=/usr/local/cuda -DUSE_CUDNN=1 -DUSE_MKLDNN=1 -DCMAKE_BUILD_TYPE=Release -GNinja ..

# RUN cd mxnet/scala-package && \
#         mvn install

# RUN cd mxnet/contrib/clojure-package && \
#         lein install
        

WORKDIR /opt/app

COPY deps.edn .
RUN clojure -A:core -Stree

EXPOSE 8080
EXPOSE 7888

CMD ["bash","c", "dev"]