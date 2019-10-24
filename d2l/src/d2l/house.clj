(ns d2l.house
  (:require [clojure.reflect :refer [reflect]]
            [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.kvstore :as kvstore]
            [org.apache.clojure-mxnet.kvstore-server :as kvstore-server]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.resource-scope :as resource-scope]
            [org.apache.clojure-mxnet.ndarray :as nd]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as shape])
  (:gen-class))

(def data-dir "./tmp/data/house/")
(def model-prefix "tmp/model/house/test")

#_(sh "bash" "-c" "mkdir -p tmp/data/house" :dir "/opt/app")
#_(sh "bash" "-c" "mkdir -p tmp/model/house" :dir "/opt/app")

#_(when-not  (.exists (io/file (str data-dir "train.csv")))
    (do (:exit (sh "bash" "-c" "bash bin/data.sh house" :dir "/opt/app"))))

(def batch-size 10) ;; the batch size
(def optimizer (optimizer/sgd {:learning-rate 0.01 :momentum 0.0}))
(def eval-metric (eval-metric/accuracy))
(def num-epoch 3) ;; the number of training epochs
(def kvstore "local") ;; the kvstore type
;;; Note to run distributed you might need to complile the engine with an option set
(def role "worker") ;; scheduler/server/worker
(def scheduler-host nil) ;; scheduler hostame/ ip address
(def scheduler-port 0) ;; scheduler port
(def num-workers 1) ;; # of workers
(def num-servers 1) ;; # of servers