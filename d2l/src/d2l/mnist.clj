(ns d2l.mnist
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

(def opts
  {:mnist.dir/shell "/opt/app/"
   :mnist.dir/mnist "/opt/app/tmp/data/mnist/"
   :mnist.dir/fashion-mnist "/opt/app/tmp/data/fashion-mnist/"
   :mnist.prefix/mnist "./tmp/data/mnist/test"
   :mnist.prefix/fashion-mnist "tmp/data/fashion-mnist/test"
   })

(defn data-dir
  [{:mnist.dir/keys [mnist fashion-mnist]}]
  mnist
  #_fashion-mnist)

(defn model-prefix
  [{:mnist.prefix/keys [mnist fashion-mnist]}]
  mnist
  #_fashion-mnist)

(defn bash-script-fetch-mnist
  [{:mnist.dir/keys [mnist]}]
  (format "
  DIR=%s
  mkdir -p $DIR
  cd $DIR

  wget http://data.mxnet.io/mxnet/data/mnist.zip 
  unzip -u mnist.zip
  " mnist))

(defn bash-script-fetch-fashion-mnist
  [{:mnist.dir/keys [fashion-mnist]}]
  (format "
  DIR=%s
  mkdir -p $DIR
  cd $DIR

  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
  wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

  gunzip *.gz
  " fashion-mnist))

(defn fetch-mnist
  [{:mnist.dir/keys [shell] :as opts}]
  (sh "bash" "-c" (bash-script-fetch-mnist opts) :dir shell))

(defn fetch-fashion-mnist
  [{:mnist.dir/keys [shell] :as opts}]
  (sh "bash" "-c" (bash-script-fetch-fashion-mnist opts) :dir shell))

#_(.exists (io/file (str (data-dir opts) "t10k-labels-idx1-ubyte")))
#_(fetch-mnist opts)
#_(fetch-fashion-mnist opts)

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

#_(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    (sym/activation "relu1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden 64})
    (sym/activation "relu2" {:data data :act-type "relu"})
    (sym/fully-connected "fc3" {:data data :num-hidden 10})
    (sym/softmax-output "softmax" {:data data})))

(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 256})
    (sym/activation "relu1" {:data data :act-type "relu"})
    (sym/fully-connected "fc3" {:data data :num-hidden 10})
    (sym/softmax-output "softmax" {:data data})))


(defn train-data []
  (mx-io/mnist-iter {:image (str (data-dir opts) "train-images-idx3-ubyte")
                     :label (str (data-dir opts) "train-labels-idx1-ubyte")
                     :label-name "softmax_label"
                     :input-shape [784]
                     :batch-size batch-size
                     :shuffle true
                     :flat true
                     :silent false
                     :seed 10
                     :num-parts num-workers
                     :part-index 0}))

(defn eval-data []
  (mx-io/mnist-iter {:image (str (data-dir opts) "t10k-images-idx3-ubyte")
                     :label (str (data-dir opts) "t10k-labels-idx1-ubyte")
                     :input-shape [784]
                     :batch-size batch-size
                     :flat true
                     :silent false
                     :num-parts num-workers
                     :part-index 0}))

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
             (m/save-checkpoint {:prefix (model-prefix opts) :epoch _num-epoch}))
         (println "Finish fit"))))))


#_(time (start [(context/cpu)]))

(def labels ["t-shirt" "trouser" "pullover" "dress" "coat"
             "sandal", "shirt", "sneaker", "bag", "ankle boot"])

(defn label->name
  [idx]
  (get labels idx))

#_(label->name 3)

