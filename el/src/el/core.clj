(ns el.core
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [clojure.xml]
            [pad.prn.core :refer [linst]]
            [pad.coll.core :refer [contained?]]
            [pad.io.core :refer [read-nth-line count-lines]]
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
            [org.apache.clojure-mxnet.shape :as shape]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.visualization :as viz])
  (:gen-class))

#_(:out (sh "bash" "-c" "bash bin/data.sh export_bert" :dir "/opt/app"))
#_(:out (sh "bash" "-c" "bash bin/data.sh wiki_sample" :dir "/opt/app"))


(def wiki-sample-dir "./tmp/data/wiki-sample/")
(def wiki-sample-file "enwiki-20191101-pages-articles1.xml-p10p30302")

(comment 
  
  (def data-raw (slurp (str wiki-sample-dir wiki-sample-file)))
  (def data-xml (clojure.xml/parse data-raw))
  
  ;
  )