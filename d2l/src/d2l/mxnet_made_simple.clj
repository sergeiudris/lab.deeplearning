; https://arthurcaillau.com/mxnet-made-simple-ndarrays-api/
(ns d2l.mxnet-made-simple
  (:require [clojure.reflect :refer [reflect]]
            [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [clojure.data.csv :refer [read-csv]]
            [clojure.java.io :as io]
            [pad.coll.core :refer [contained?]]
            [pad.io.core :refer [read-nth-line read-n-lines count-lines
                                 pprn-n-lines]]
            [pad.core :refer [str-float? str>>float resolve-var]]
            [pad.math.core :refer [vec-standard-deviation-2
                                   scalar-subtract elwise-divide
                                   vec-mean scalar-divide
                                   mk-one-hot-vec std]]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.kvstore :as kvstore]
            [org.apache.clojure-mxnet.kvstore-server :as kvstore-server]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.lr-scheduler :as lr-scheduler]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.resource-scope :as resource-scope]
            [org.apache.clojure-mxnet.ndarray :as nd]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as shape])
  (:gen-class))

(def sample-size 1000)
(def train-size 800)
(def valid-size (- sample-size train-size))
(def feature-count 100)
(def category-count 10)
(def batch-size 10)

(def X
  (random/uniform 0 1 [sample-size feature-count]))

(def Y
  (-> sample-size
      (repeatedly #(rand-int category-count))
      (nd/array [sample-size])))

(nd/shape-vec X) ; [1000 100]
(take 5 (nd/->vec X)) ; (0.5488135 0.5928446 0.71518934 0.84426576 0.60276335)
(nd/shape-vec Y) ; [1000]
(take 5 (nd/->vec Y)) ; (3.0 1.0 0.0 0.0 3.0)

(def X-train
  (nd/crop X
           (shape/->shape [0 0])
           (shape/->shape [train-size feature-count])))

(def X-valid
  (nd/crop X
           (shape/->shape [train-size 0])
           (shape/->shape [sample-size feature-count])))

(def Y-train
  (nd/crop Y
           (shape/->shape [0])
           (shape/->shape [train-size])))

(def Y-valid
  (nd/crop Y
           (shape/->shape [train-size])
           (shape/->shape [sample-size])))

(defn get-symbol
  []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    (sym/activation "act1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden category-count})
    (sym/softmax-output "softmax" {:data data})))

(defn train!
  [model-module]
  (-> model-module
      (m/bind {:data-shapes (mx-io/provide-data train-iter)
               :label-shapes (mx-io/provide-label valid-iter)})
      (m/init-params {:initializer (initializer/xavier)})
      (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1})})
      (m/fit {:train-data train-iter
              :eval-data valid-iter
              :num-epoch 50})))

(comment
  (def train-iter
    (mx-io/ndarray-iter [X-train]
                        {:label-name "softmax_label"
                         :label [Y-train]
                         :data-batch-size batch-size}))

  (def valid-iter
    (mx-io/ndarray-iter [X-valid]
                        {:label-name "softmax_label"
                         :label [Y-valid]
                         :data-batch-size batch-size}))

  (def model-module (m/module (get-symbol)))

  (mx-io/reset train-iter)
  (mx-io/reset valid-iter)
  (train! model-module)

  (m/score model-module {:eval-data valid-iter
                         :eval-metric (eval-metric/accuracy)})
  
  
  ;
  )




