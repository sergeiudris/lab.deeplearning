(ns d2l.viz
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.visualization :as viz]))

(def data-dir "./tmp/data/viz/")

(defn load-data!
  []
  (when-not  (.exists (io/file (str data-dir "vgg16-symbol.json")))
    (do
      (:exit (sh "bash" "-c" "bash bin/data.sh viz" :dir "/opt/app")))))

#_(load-data!)

(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
             model-sym
             {"data" input-data-shape}
             {:title model-name
              :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot model-name path)))

(comment

  (def vgg16-mod
    "VGG16 Module"
    (m/load-checkpoint {:prefix (str data-dir "/vgg16") :epoch 0}))

  (def resnet18-mod
    "Resnet18 Module"
    (m/load-checkpoint {:prefix (str data-dir "/resnet-18") :epoch 0}))


  (render-model! {:model-name "vgg16"
                  :model-sym (m/symbol vgg16-mod)
                  :input-data-shape [1 3 244 244]
                  :path data-dir})

  (render-model! {:model-name "resnet18"
                  :model-sym (m/symbol resnet18-mod)
                  :input-data-shape [1 3 244 244]
                  :path data-dir})
  

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
  
  (render-model! {:model-name "lenet"
                  :model-sym (get-symbol)
                  :input-data-shape [1 3 28 28]
                  :path data-dir})
  
  
  ;
  )
