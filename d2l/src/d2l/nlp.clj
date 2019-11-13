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
            [pad.math.core :refer [vec-standard-deviation-2
                                   scalar-subtract elwise-divide
                                   vec-mean scalar-divide
                                   mk-one-hot-vec std vec-normalize]]
            [pad.mxnet.bert :as bert]
            [pad.mxnet.core :refer [read-glove! glove-path]]
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

#_(def glove (-> (glove-path glove-dir 50) (read-glove!)))
#_(def glove-embeddings (:token-to-embedding glove))
#_(def glove-idxs (:idx-to-token glove))

#_(count glove-idxs)
#_(->> glove-idxs (take 2))
#_(get glove-idxs 242662)

#_(def glove-normalized (->> glove-embeddings
                             (seq)
                             (reduce (fn [a [k v]]
                                       (assoc a k (vec-normalize v))) {})))

#_(def glove-nd (nd/array
                 (mapcat second (seq glove-normalized))
                 [(count glove-embeddings) (-> glove-embeddings (first) (second) (count))]))

(defn glove-dot
  [glove-nd v]
  (nd/dot glove-nd v))

#_(def glove-arrayed (->> glove-normalized
                          (seq)
                          (reduce (fn [a [k v]]
                                    (assoc a k (nd/array v [(count v)]))) {})))

#_(def w-baby (get glove-arrayed "baby"))

#_(def w-dot (glove-dot glove-nd w-baby))

#_(def topk (-> (ndapi/topk {:data w-dot :axis 0 :k 5 :ret-typ "indices"}) (nd/->vec)))
#_(def topk (nd/topk w-dot 0  5 "indices"))

#_(->> topk (mapv #(get glove-idxs (int %))))

#_(into {} [[1 1] [2 2]])
#_(vec-normalize [1 2 3])


