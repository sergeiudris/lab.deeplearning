FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y \
            sudo  \
            curl  \
            wget  \
            git-core  \
            gpg  \
            rlwrap  \
            software-properties-common  \
            unzip curl net-tools lsof

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk

ENV CLOJURE_SCRIPT_VERSION=linux-install-1.10.1.466.sh

RUN curl -O https://download.clojure.org/install/$CLOJURE_SCRIPT_VERSION && \
    chmod +x $CLOJURE_SCRIPT_VERSION && \
    sudo ./$CLOJURE_SCRIPT_VERSION

RUN sudo add-apt-repository ppa:timsc/opencv-3.4 && \
    apt-get update && \
    apt-get install -y  \
        libopencv-imgcodecs3.4 \
        libopenblas-base \
        libatlas3-base \
        libcurl3 

WORKDIR /opt/app

COPY deps.edn .
RUN clojure -A:core -Stree

EXPOSE 8080
EXPOSE 7888

CMD ["bash","c", "dev"]