## links

- https://github.com/NVIDIA/nvidia-docker
    - in 'software&updates' select nvidia driver as active
    - create password
    - when laptop restarts, blue screen appears
    - select 'enroll MOK'
    - not sure next
    - enter password 
- https://github.com/NVIDIA/nvidia-docker/wiki/Driver-containers-(Beta)#quickstart
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
    - notes:
        - the issue is likely caused by 4G GPU RAM being not enough
        - in general, it seems you need at least 8-10 GB
        - when training with BERT on CPU , RAM usage is ~10 GB with batch-size 32


#### tfjs on node

- https://www.tensorflow.org/js/guide/nodejs
    - as guide clarifies, single threaded nature of node makes tfjs-node unequal to tf itself or mxnet