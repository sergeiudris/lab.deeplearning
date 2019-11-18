(ns d2l.viz
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.visualization :as viz]))

(def opts
  {:viz.dir/shell "/opt/app/"
   :viz.dir/target "/opt/app/tmp/data/viz/"})

(defn data-dir
  [{:viz.dir/keys [target]}]
  target)

(defn script-fetch-viz
  [{:viz.dir/keys [target]}]
  (format "
  DIR=%s
  mkdir -p $DIR
  cd $DIR

  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params

  wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json
  wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params
  " target))

(defn fetch-viz
  [{:viz.dir/keys [shell] :as opts}]
  (sh "bash" "-c" (script-fetch-viz opts) :dir shell))

#_(.exists (io/file (str data-dir "vgg16-symbol.json")))
#_(:exit (fetch-viz opts))

(comment

  (def vgg16-mod
    "VGG16 Module"
    (m/load-checkpoint {:prefix (str (data-dir opts) "/vgg16") :epoch 0}))

  (def resnet18-mod
    "Resnet18 Module"
    (m/load-checkpoint {:prefix (str (data-dir opts) "/resnet-18") :epoch 0}))


  (-> (viz/plot-network
       (m/symbol vgg16-mod)
       {"data" [1 3 244 244]}
       {:title "vgg16"
        :node-attrs {:shape "oval" :fixedsize "false"}})
      (viz/render "vgg16" (data-dir opts)))

  (-> (viz/plot-network
       (m/symbol resnet18-mod)
       {"data" [1 3 244 244]}
       {:title "resnet18"
        :node-attrs {:shape "oval" :fixedsize "false"}})
      (viz/render "resnet18" (data-dir opts)))

  ;
  )


(defn get-symbol
  "Return LeNet Symbol

  Input data shape [`batch-size` `channels` 28 28]
  Output data shape [`batch-size 10]"
  []
  (as-> (sym/variable "data") data

    ;; First `convolution` layer
    (sym/convolution "conv1" {:data data :kernel [5 5] :num-filter 20})
    (sym/activation "tanh1" {:data data :act-type "tanh"})
    (sym/pooling "pool1" {:data data :pool-type "max" :kernel [2 2] :stride [2 2]})

    ;; Second `convolution` layer
    (sym/convolution "conv2" {:data data :kernel [5 5] :num-filter 50})
    (sym/activation "tanh2" {:data data :act-type "tanh"})
    (sym/pooling "pool2" {:data data :pool-type "max" :kernel [2 2] :stride [2 2]})

    ;; Flattening before the Fully Connected Layers
    (sym/flatten "flatten" {:data data})

    ;; First `fully-connected` layer
    (sym/fully-connected "fc1" {:data data :num-hidden 500})
    (sym/activation "tanh3" {:data data :act-type "tanh"})

    ;; Second `fully-connected` layer
    (sym/fully-connected "fc2" {:data data :num-hidden 10})

    ;; Softmax Loss
    (sym/softmax-output "softmax" {:data data})))


(comment

  (-> (viz/plot-network
       (get-symbol)
       {"data" [1 3 28 28]}
       {:title "lenet"
        :node-attrs {:shape "oval" :fixedsize "false"}})
      (viz/render "lenet" (data-dir opts)))


  ;
  )
