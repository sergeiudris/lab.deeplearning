- d2l
  - book "Dive into Deep Learning"
    - https://d2l.ai/
  - issues
    - 

- GPU
  - nvidia setup
    - https://github.com/NVIDIA/nvidia-docker
      - in 'software&updates' select nvidia driver as active
      - create password
      - when laptop restarts, blue screen appears
      - select 'enroll MOK'
      - not sure next
      - enter password
    - https://github.com/NVIDIA/nvidia-docker/wiki/Driver-containers-(Beta)#quickstart
  - issues
    - CuDNNBatchNorm
      - caused by importing other examples into d2l app which had mxnet-linux-cpu deps instead of gpu
    - cudaErrorCudartUnloading: CUDA: CUDA driver version is insufficient for CUDA runtime version
      - https://stackoverflow.com/questions/41409842/ubuntu-16-04-cuda-8-cuda-driver-version-is-insufficient-for-cuda-runtime-vers
      - driver compatibility table
        - https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
    - docker run --rm --runtime=nvidia nvidia/cuda:9.2-base nvidia-smidocker
      - Error response from daemon: Unknown runtime specified nvidia
        - https://github.com/docker/compose/issues/6691
    - BERT model:  cudaMalloc failed: out of memory
      - https://github.com/apache/incubator-mxnet/issues/4224
      - notes
        - the issue is likely caused by 4G GPU RAM being not enough
        - in general, it seems you need at least 8-10 GB
          - when training with BERT on CPU , RAM usage is ~10 GB with batch-size 32
    - docker-compose --gpus
      - https://github.com/docker/compose/issues/6691
      - use docker run --gpus directly while docker-compose does not support --gpus flag
    
- tfjs
  - https://www.tensorflow.org/js/guide/nodejs
  - value
    - as guide clarifies, single threaded nature of node makes tfjs-node unequal to tf itself or mxnet

- gvm
    - issues
      - Call path from entry point to clojure.spec.gen.alpha$dynaload$fn__2628.invoke():
        - https://github.com/oracle/graal/issues/1266
        - https://github.com/oracle/graal/issues/1681

- dl4j
  - issues
    - network issue, host blob.deeplearning4j.org cannot be not resolved
      - http://blob.deeplearning4j.org/datasets/iris.dat doesn't resolve even in Opera with vpn
      - probably a geographical issue
      - resolved
          - https://github.com/eclipse/deeplearning4j-examples/issues/924
    - Execution error (DL4JInvalidInputException) at org.deeplearning4j.nn.layers.BaseLayer/preOutputWithPreNorm (BaseLayer.java:306).
      Input that is not a matrix; expected matrix (rank 2), got rank 1 array with shape [784]. Missing preprocessor or wrong input type?
        - https://github.com/eclipse/deeplearning4j/issues/3112

- lein-virgil
  - issues
    - :reloading .. :error-when-loading ..
      - resolved
          - rm -rf target


- text similairty
    - BERT word and sentence embeddings
        - https://github.com/google-research/bert/issues/261
            - https://github.com/hanxiao/bert-as-service/#q-what-are-the-available-pooling-strategies
            - https://github.com/imgarylai/bert-embedding
        - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
            - https://arxiv.org/abs/1908.10084
        - https://stackoverflow.com/questions/58168936/bert-sentence-embeddings
        - News Topic Similarity Measure using Pretrained BERT Model
            - https://medium.com/the-artificial-impostor/news-topic-similarity-measure-using-pretrained-bert-model-1dbfe6a66f1d
        - https://github.com/Separius/awesome-sentence-embedding
        - https://datascience.stackexchange.com/questions/62658/how-to-get-sentence-embedding-using-bert
    - Text Similarities : Estimate the degree of similarity between two texts
        - https://medium.com/@adriensieg/text-similarities-da019229c894
        - https://github.com/adsieg/text_similarity
    - GluonNLP — Deep Learning Toolkit for Natural Language Processing
        - https://medium.com/apache-mxnet/gluonnlp-deep-learning-toolkit-for-natural-language-processing-98e684131c8a
    
    
- git search
    - https://git-scm.com/book/en/v2/Git-Tools-Searching
    - https://stackoverflow.com/questions/4468361/search-all-of-git-history-for-a-string
    - https://github.com/sourcegraph/sourcegraph

- clj protocols
    - explained
        - https://stackoverflow.com/questions/4509782/simple-explanation-of-clojure-protocols
        - https://stackoverflow.com/questions/37058268/what-is-reify-in-clojure
    - examples
        - https://stackoverflow.com/questions/38573470/make-a-class-that-extends-a-class-with-overriding-in-clojure
        - https://stackoverflow.com/questions/3057034/adding-fields-to-a-proxied-class-in-clojure
    