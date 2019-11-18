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
            [pad.ml.nlp :refer [build-vocab]]
            [pad.dataset.glove :refer [read-glove! glove-filepath]]
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

#_(:exit (sh "bash" "-c" "bash bin/data.sh text8" :dir "/opt/app"))

(def opts {:glove.dir/shell "/opt/app/"
           :glove.dir/target "/opt/app/tmp/data/glove/"
           :glove/embedding-size 50
           :text8.dir/shell "/opt/app/"
           :text8.dir/target "/opt/app/tmp/data/text8/"
           })

(defn word>>slice
  [word index mx]
  (-> (nd/slice mx (get index word))
      (nd/reshape [-1 1])))

(comment

  (nd/concatenate [(nd/array [1 2] [2]) (nd/array [1 2] [2])])
  (def v (get glove-embeddings "matrix"))

  (do
    (def glove (-> (glove-filepath opts) (read-glove!)))
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

  (def word1 (word>>slice "do" glove-to-idx mx))
  (def word2 (word>>slice "did" glove-to-idx mx))
  (def word3 (word>>slice "go" glove-to-idx mx))

  (def word-diff (nd/+ word3 (nd/- word2 word1)))
  (def word-diff-dot-prod (-> (nd/dot mx-norm word-diff)
                              (nd/reshape [(count glove-vec)])))
  (def topk (-> (ndapi/topk {:data word-diff-dot-prod :axis 0 :k 1 :ret-typ "indices"})))
  (->> topk (nd/->vec) (mapv #(get glove-to-token (int %))))

  ;
  )

(defn bash-script-fetch-text8
  [{:text8.dir/keys [target]}]
  (format "
  DIR=%s
  mkdir -p $DIR
  cd $DIR
        
  wget https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/large_text_compression_benchmark/text8-6c70299b.zip
  unzip *.zip
  " target))

(defn fetch-text8
  [{:text8.dir/keys [shell] :as opts}]
  (sh "bash" "-c" (bash-script-fetch-text8 opts) :dir shell))

(defn text8-dir
  [{:text8.dir/keys [target]}]
  target)


(comment

  (def text8-raw (slurp (str (text8-dir opts) "text8")))
  (def text8 (-> text8-raw (string/split  #"\s") (rest)))
  (def sens (partition 10000 10000 [] text8))
  (count sens)
  (take 10 (nth sens 1))
  (take 10 (nth sens 2))
  (take 10 (nth sens 3))

  (def vocab (build-vocab text8))
  (->> vocab :frequencies (take 10))
  (->> vocab :frequencies-sorted (take 10))
  (-> vocab :indexes (get "the"))
  (def indices (into {} (filter (fn [[word idx]]
                                  (>= (get (:frequencies vocab) word) 5)) (:indexes vocab))))
  (->> indices (sort-by second <) (take 10))

  (def sens-tokens (map #(keep (fn [token]
                                 (get indices token)) %) sens))
  (-> sens-tokens (nth 1) (count)) ; 9895 9858 9926
  (->> (nth sens-tokens 1) (take 5)) ; 18228 17322 36981 4 1754

  (partition  2 2 [] [1 2 3 4 5 6 7])


  ;
  )

