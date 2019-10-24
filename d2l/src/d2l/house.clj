(ns d2l.house
  (:require [clojure.reflect :refer [reflect]]
            [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [clojure.data.csv :refer [read-csv]]
            [tools.io.core :refer [read-nth-line count-lines]]
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


#_(read-nth-line (str data-dir "train.csv") 1)
#_(read-nth-line (str data-dir "test.csv") 1)


(defn read-csv-rows
  [filename]
  (with-open [reader (io/reader filename)]
    (->> (read-csv reader)
         (rest)
         (vec))))

(defn read-labels
  [filename]
  (with-open [reader (io/reader filename)]
    (->> (read-csv reader)
         (rest)
         (mapv #(nth % 80)))))

(defn read-column
  [filename idx]
  (with-open [reader (io/reader filename)]
    (->> (read-csv reader)
         (rest)
         (mapv #(nth % idx)))))
#_(def attrs
    (->
     (read-nth-line (str data-dir "train.csv") 1)
     (str/split  #",")))
#_(def features (-> attrs (rest) (vec)))
#_(def train-dataset
    (read-csv-rows (str data-dir "train.csv")))
#_(def test-dataset
    (read-csv-rows (str data-dir "test.csv")))
#_(def train-labels (read-labels (str data-dir "train.csv")))
#_(count attrs) ; 81
#_(count features) ; 80
#_(count train-dataset) ; 1460
#_(count test-dataset) ; 1459
#_(count train-labels) ; 1460

#_(take 10 train-labels)



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

(def envs (cond-> {"DMLC_ROLE" role}
            scheduler-host (merge {"DMLC_PS_ROOT_URI" scheduler-host
                                   "DMLC_PS_ROOT_PORT" (str scheduler-port)
                                   "DMLC_NUM_WORKER" (str num-workers)
                                   "DMLC_NUM_SERVER" (str num-servers)})))



(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    (sym/activation "relu1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden 64})
    (sym/activation "relu2" {:data data :act-type "relu"})
    (sym/fully-connected "fc3" {:data data :num-hidden 10})
    (sym/softmax-output "softmax" {:data data})))

(defn train-data []
  #_(mx-io/ndarray-iter [(ndarray/array (flatten train-data)
                                      [(count train-data) sent-len])]
                      {:label [(ndarray/array (flatten labels)
                                              [(count labels) sent-len])]
                       :label-name "softmax_label"
                       :data-batch-size batch-size
                       :last-batch-handle "pad"}))

(defn eval-data []
  #_(mx-io/ndarray-iter [(get-in shuffled [:test :data])]
                      {:label [(get-in  shuffled [:test :label])]
                       :label-name "softmax_label"
                       :data-batch-size batch-size
                       :last-batch-handle "pad"}))

(defn start
  ([devs] (start devs num-epoch))
  ([devs _num-epoch]
   (when scheduler-host
     (println "Initing PS enviornments with " envs)
     (kvstore-server/init envs))

   (if (not= "worker" role)
     (do
       (println "Start KVStoreServer for scheduler and servers")
       (kvstore-server/start))
     (do
       (println "Starting Training of MNIST ....")
       (println "Running with context devices of" devs)
       (resource-scope/with-let [_mod (m/module (get-symbol) {:contexts devs})]
         (-> _mod
             (m/fit {:train-data (train-data)
                     :eval-data (eval-data)
                     :num-epoch _num-epoch
                     :fit-params (m/fit-params {:kvstore kvstore
                                                :optimizer optimizer
                                                :eval-metric eval-metric})})
             (m/save-checkpoint {:prefix model-prefix :epoch _num-epoch}))
         (println "Finish fit"))))))


#_(time (start [(context/cpu)]))