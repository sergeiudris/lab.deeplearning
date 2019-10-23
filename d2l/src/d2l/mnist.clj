(ns d2l.mnist
  (:require [clojure.reflect :refer [reflect]]
            [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.ndarray :as nd]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as shape]))


(def data-dir "./.data")

(defn file-exists?
  [name]
  (.exists (io/file name)))

#_(when-not  (file-exists? (str data-dir "/t10k-labels-idx1-ubyte"))
    (do (:exit (sh "bash" "-c" "bash bin/load_fashion_mnist.sh" :dir "/opt/app"))))
