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
            graphviz \
            zlib1g-dev gcc libc6-dev

# RUN apt-get update && \
#     apt-get install -y openjdk-8-jdk

WORKDIR /tmp


# https://github.com/arjones/docker-graalvm/blob/master/Dockerfile

#  graalvm-ce-java8-19.3.0
ENV GRAALVM_VERSION=19.3.0
ENV SUFFIX_URL=java8-linux-amd64-${GRAALVM_VERSION}
ENV SUFFIX_DIR=java8-${GRAALVM_VERSION}

# RUN curl -Ls "https://github.com/oracle/graal/releases/download/vm-${GRAALVM_VERSION}/graalvm-ce-${GRAALVM_VERSION}-linux-amd64.tar.gz" | \
RUN curl -Ls "https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-19.3.0/graalvm-ce-${SUFFIX_URL}.tar.gz" | \
    tar zx -C /usr/local/ && \
    ls -l /usr/local/ && \
    rm -f /usr/local/graalvm-ce-${SUFFIX_DIR}/src.zip && \
    ln -s /usr/local/graalvm-ce-${SUFFIX_DIR} /usr/local/graalvm && \
    rm -fr /var/lib/apt
ENV PATH $PATH:/usr/local/graalvm/bin
RUN gu install native-image && gu install python

# ENV GRAAL_ARCHIVE=graalvm-ce-java8-linux-amd64-19.3.0.tar.gz

# RUN wget -q https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-19.3.0/$GRAAL_ARCHIVE 
# RUN wget -q https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-19.3.0/graalvm-ce-java11-linux-amd64-19.3.0.tar.gz


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

RUN apt-get update && \
    apt-get install -y python3-pip  && \
    pip3 install gluonnlp mxnet


WORKDIR /opt/app

COPY lab/deps.edn .
RUN clojure -A:core -Stree

EXPOSE 8080
EXPOSE 7888

CMD ["bash","c", "dev"]