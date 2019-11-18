(ns el.core
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [pad.prn.core :refer [linst]]
            [pad.coll.core :refer [contained?]]
            [pad.io.core :refer [read-nth-line count-lines]]
            [pad.core :refer [str-float? str>>float resolve-var]]
            [pad.math.core :refer [vec-standard-deviation-2
                                   scalar-subtract elwise-divide
                                   vec-mean scalar-divide
                                   mk-one-hot-vec std]]
            [pad.dataset.cmu :refer [read-metadata! read-summaries!
                                     data>>joined fetch-cmu]]
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
            [org.apache.clojure-mxnet.visualization :as viz]

            [pad.dataset.cmu :refer [read-summaries!]]
            [pad.dataset.wiki])
  (:gen-class))

(def opts
  {:cmu.dir/shell "/opt/app/"
   :cmu.dir/target "/opt/app/tmp/data/cmu/"})

#_(fetch-cmu opts)
#_(sh "bash" "-c" "rm -rf tmp/data/cmu" :dir (opts :cmu.dir/shell))

(comment


  (do
    (def mdata (read-metadata! opts))
    (def summs (read-summaries! opts))
    (def data (data>>joined mdata summs))
    (def data-sorted (->> data (sort-by :box-office >)))
    (def data-sorted-map (->> data-sorted (reduce #(assoc %1 (:id-wiki %2) %2) {}))))

  (count data-sorted)
  (->> data-sorted (map :box-office) (take 10))
  (->> data-sorted (take 20) (map #(select-keys % [:id-wiki :name :box-office])))
  (-> data-sorted-map (get "174251") (select-keys [:id-wiki :name :box-office]))

  ;
  )