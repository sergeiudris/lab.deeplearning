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
    apt-get install -y openjdk-11-jdk

ENV CLOJURE_SCRIPT_VERSION=linux-install-1.10.1.466.sh

RUN curl -O https://download.clojure.org/install/$CLOJURE_SCRIPT_VERSION && \
    chmod +x $CLOJURE_SCRIPT_VERSION && \
    sudo ./$CLOJURE_SCRIPT_VERSION

WORKDIR /tmp

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

RUN sudo add-apt-repository ppa:timsc/opencv-3.4 && \
    apt-get update && \
    apt-get install -y  \
        libopencv-imgcodecs3.4 \
        libopenblas-base \
        libatlas3-base \
        libcurl3 

# RUN apt-get install -y pacman && \
#     wget https://archive.archlinux.org/packages/c/cuda/cuda-9.0.176-4-x86_64.pkg.tar.xz && \
#     pacman -U cuda-9.0.176-4-x86_64.pkg.tar.xz

WORKDIR /opt/app

COPY src src
COPY project.clj project.clj
RUN lein install

COPY examples/bert/project.clj examples/bert/project.clj 
RUN cd examples/bert && lein deps

EXPOSE 8080
EXPOSE 7888


CMD ["bash","c", "dev"]