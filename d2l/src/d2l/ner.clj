(ns d2l.ner
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


(def data-dir "./tmp/data/ner/")
(def model-prefix "tmp/model/ner/test")

(defn readv-file!
  [filename]
  (with-open [reader (io/reader filename)]
    (let [data (read-csv reader)]
      (vec data))))

#_(def dset (readv-file! (str data-dir "ner_dataset.csv")))
#_(take 100 dset)
#_(read-nth-line (str data-dir "ner_dataset.csv") 5 )
#_(read-n-lines (str data-dir "ner_dataset.csv"))
#_(pprn-n-lines (str data-dir "ner_dataset.csv") 0 10)