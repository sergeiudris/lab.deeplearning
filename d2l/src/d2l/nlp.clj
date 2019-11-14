(ns d2l.nlp
  (:require [clojure.reflect :refer [reflect]]
            [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [clojure.data.csv :refer [read-csv]]
            [clojure.data.xml :as xml]
            [clojure.zip :as zip]
            [clojure.xml]
            [cheshire.core :as json]
            [pad.prn.core :refer [linst]]
            [pad.coll.core :refer [contained?]]
            [pad.io.core :refer [read-nth-line count-lines]]
            [pad.core :refer [str-float? str>>float resolve-var]]
            [pad.mxnet.bert :as bert]
            [pad.mxnet.core :refer [read-glove! glove-path normalize normalize-row]]
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
            [org.apache.clojure-mxnet.ndarray-api :as ndapi]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as shape]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.visualization :as viz])
  (:gen-class))

(def data-dir "./tmp/data/nlp/")
(def glove-dir "./tmp/data/glove/")


(defn load-data!
  []
  (:exit (sh "bash" "-c" "bash bin/data.sh glove" :dir "/opt/app")))

#_(load-data!)

(defn word>>slice
  [word index mx]
  (-> (nd/slice mx (get index word))
      (nd/reshape [-1 1])))

(comment

  (nd/concatenate [(nd/array [1 2] [2]) (nd/array [1 2] [2])])
  (def v (get glove-embeddings "matrix"))

  (do
    (def glove (-> (glove-path glove-dir 50) (read-glove!)))
    (def glove-vec (:vec glove))
    (def glove-to-embedding (:token-to-embedding glove))
    (def glove-to-token (:idx-to-token glove))
    (def glove-to-idx (:token-to-idx glove))

    (first glove-vec)
    (get glove-to-embedding "the")
    (get glove-to-token 0)
    (get glove-to-idx "the")

    (def mx (nd/array
             (mapcat second glove-vec)
             [(count glove-vec) (-> glove-vec (first) (second) (count))]))
    (def mx-norm (ndapi/l2-normalization {:data mx :eps 1E-10})))

  ; word similarity

  (def word (word>>slice "computers" glove-to-idx mx))
  (def word-dot-prod (-> (nd/dot mx-norm word)
                         (nd/reshape [(count glove-vec)])))
  (def topk (-> (ndapi/topk {:data word-dot-prod :axis 0 :k 10 :ret-typ "indices"})))
  (->> topk (nd/->vec) (mapv #(get glove-to-token (int %))))

  ; word analogy

  (def word1 (word>>slice "man" glove-to-idx mx))
  (def word2 (word>>slice "woman" glove-to-idx mx))
  (def word3 (word>>slice "son" glove-to-idx mx))

  (def word-diff (nd/+ word3 (nd/- word2 word1)))
  (def word-diff-dot-prod (-> (nd/dot mx-norm word-diff)
                              (nd/reshape [(count glove-vec)])))
  (def topk (-> (ndapi/topk {:data word-diff-dot-prod :axis 0 :k 1 :ret-typ "indices"})))
  (->> topk (nd/->vec) (mapv #(get glove-to-token (int %))))




  ;
  )

