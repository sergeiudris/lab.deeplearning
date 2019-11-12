# FROM ubuntu:18.04
# FROM nvidia/cuda:9.2-devel-ubuntu18.04
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

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

ENV LEIN_VERSION=2.9.1
ENV LEIN_INSTALL=/usr/local/bin/

# Download the whole repo as an archive
RUN mkdir -p $LEIN_INSTALL \
  && wget -q https://raw.githubusercontent.com/technomancy/leiningen/$LEIN_VERSION/bin/lein-pkg \
  # && echo "Comparing lein-pkg checksum ..." \
  # && echo "019faa5f91a463bf9742c3634ee32fb3db8c47f0 *lein-pkg" | sha1sum -c - \
  && mv lein-pkg $LEIN_INSTALL/lein \
  && chmod 0755 $LEIN_INSTALL/lein \
  && wget -q https://github.com/technomancy/leiningen/releases/download/$LEIN_VERSION/leiningen-$LEIN_VERSION-standalone.zip \
  && wget -q https://github.com/technomancy/leiningen/releases/download/$LEIN_VERSION/leiningen-$LEIN_VERSION-standalone.zip.asc \
  && rm leiningen-$LEIN_VERSION-standalone.zip.asc \
  && mkdir -p /usr/share/java \
  && mv leiningen-$LEIN_VERSION-standalone.zip /usr/share/java/leiningen-$LEIN_VERSION-standalone.jar

ENV PATH=$PATH:$LEIN_INSTALL
ENV LEIN_ROOT 1

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
        

# installing clojure pkg from source

# RUN git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet && \
#         cd mxnet && \
#         git checkout c81535ce33a9bb167662af53181f9a42ed5b4bc4 && \
#         git submodule update --init --recursive

# RUN cd mxnet/contrib/clojure-package && \
#         sed -i 's/mxnet-full_2.11/mxnet-full_2.11-linux-x86_64-gpu/g' project.clj && \
#         sed -i 's/INTERNAL/1.6.0-SNAPSHOT/g' project.clj && \
#         lein install

RUN apt-get update && \
    apt-get install -y python3-pip  && \
    pip3 install gluonnlp mxnet


WORKDIR /opt/app

COPY deps.edn .
RUN clojure -A:core -Stree

EXPOSE 8080
EXPOSE 7888

CMD ["bash","c", "dev"]