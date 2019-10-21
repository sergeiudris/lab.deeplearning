# taken from https://bitbucket.org/magnet-coop/mxnet-clj-cpu/src/master/Dockerfile
# FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

RUN set -ex; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            wget runit openjdk-8-jdk-headless sudo libopenblas-base libatlas3-base \
            libcurl3 software-properties-common unzip && \
    add-apt-repository -y ppa:timsc/opencv-3.4 && \
    apt-get update && \
    apt-get install -y libopencv-imgcodecs3.4 && \
    wget http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb && \
    dpkg -i scala-2.11.8.deb && \
    rm scala-2.11.8.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 
    # useradd --home-dir /home/magnet --create-home --shell /bin/bash --user-group magnet

COPY --from=clojure:lein-2.8.1 /usr/local/bin/lein /usr/local/bin/

COPY --from=clojure:lein-2.8.1 /usr/share/java/leiningen-2.8.1-standalone.jar /usr/share/java/

WORKDIR /opt/app

COPY src src
COPY project.clj project.clj
RUN lein install

COPY examples/bert/project.clj examples/bert/project.clj 
RUN cd examples/bert && lein deps

EXPOSE 8080
EXPOSE 7888