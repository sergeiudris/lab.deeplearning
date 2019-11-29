(ns d2l.recom
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
                                   mk-one-hot-vec std]]
            [pad.dataset.cmu :refer [read-metadata! read-summaries!
                                     data>>joined]]
            [pad.dataset.bert :refer [read-vocab-json! data>>tokened pair>>padded]]
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

(def bert-dir "/opt/app/tmp/data/bert/")
(def bert-base-prefix "static_bert_base_net")
(def bert-base-vocab-filename "vocab.json")
(def bert-exported-dir "/opt/app/tmp/data/bert-export/")

(def opts
  {:cmu.dir/shell "/opt/app/"
   :cmu.dir/target "/opt/app/tmp/data/cmu/"

   :bert.python/output-dir bert-exported-dir
   :bert.dir/from-mxnet-example bert-dir
   :bert.dir/python-scripts "/opt/app/python/bert/"
   :bert.dir/mxnet "/root/.mxnet/"})

(defn predict-equivalent
  [{:keys [predictor vocab seq-length sentence-a sentence-b]}]
  (let [pair  (pair>>padded [sentence-a
                             sentence-b]
                            vocab
                            {:seq-length seq-length})]
    (->>
     (infer/predict-with-ndarray
      predictor
      [(nd/array (-> pair :batch :idxs) [1 seq-length])
       (nd/array (-> pair :batch :token-types) [1 seq-length])
       (nd/array (-> pair :batch :valid-length) [1])])
     (first)
     (nd/->vec))))

(defn find-equivalents
  [{:keys [predictor data vocab seq-length sentence]}]
  (->> data
       (mapv (fn [v]
               (predict-equivalent {:predictor predictor
                                    :vocab vocab
                                    :seq-length seq-length
                                    :sentence-a sentence
                                    :sentence-b v})))
       (mapv (fn [v score]
               (assoc v :score (first score))) data)
       (sort-by :score >)))

#_(do
    (def mdata (read-metadata! opts))
    (def summs (read-summaries! opts))
    (def data (data>>joined mdata summs))
    (def bert-vocab (read-vocab-json! (str bert-dir bert-base-vocab-filename)))
    (def data-tokened (data>>tokened data #(:summary %)))
    (def data-filtered (->> data-tokened (filterv #(<= (-> % :tokens (count)) 254))))
    ; (def data-padded (bert/data>>padded data-filtered bert-vocab {:seq-length 512}))
    (def data-sorted (->> data-filtered (sort-by :box-office >)))
    (def seq-length 512)
    (def data-sorted-map (->> data-sorted (reduce #(assoc %1 (:id-wiki %2) %2) {}))))

#_(->> (get bert-vocab "idx_to_token") (count)) ; 30522
#_(->> (get bert-vocab "token_to_idx") (count)) ; 30522
#_(-> (get bert-vocab "token_to_idx") (get "[SEP]" ))

#_(count data-sorted)
#_(->> data-tokened (map #(count (:tokens %))) (apply max))
#_(->> data-tokened (filter #(< (-> % :tokens (count)) 128)) (count))
#_(->> data-tokened (map :box-office) (take 10))
#_(->> data-tokened
       (filter #(< (-> % :tokens (count)) 128))
       (sort-by :box-office >)
       (take 20)
       (map #(select-keys % [:name :box-office])))
#_(-> data-sorted (first) :tokens (count))
#_(->> data-sorted (take 20) (map #(select-keys % [:id-wiki :name :box-office])))
#_(-> data-sorted-map (get "161190") (select-keys [:id-wiki :name :box-office]))

(comment

  (def predictor
    (infer/create-predictor
     (infer/model-factory
      (str bert-exported-dir "regression")
      [{:name "data0" :shape [1 seq-length] :dtype dtype/FLOAT32 :layout layout/NT}
       {:name "data1" :shape [1 seq-length] :dtype dtype/FLOAT32 :layout layout/NT}
       {:name "data2" :shape [1]            :dtype dtype/FLOAT32 :layout layout/N}])
     {:epoch 0}))

  (def pair-padded (bert/pair>>padded [(get data-sorted-map "161190")
                                       (get data-sorted-map "20688153")]
                                      bert-vocab
                                      {:seq-length 512}))
  (-> pair-padded :tokens (count))
  (-> pair-padded :batch :idxs (count))
  (-> pair-padded :batch :token-types (count))
  (-> pair-padded :batch :valid-length (count))

  (def rlt (predict-equivalent {:predictor predictor
                                :seq-length seq-length
                                :vocab bert-vocab
                                :sentence-a (get data-sorted-map "20688153")
                                :sentence-b (get data-sorted-map "161190")
                                #_(get data-sorted-map "161190")
                                #_(get data-sorted-map "20688153")}))


  (def rlts (find-equivalents {:predictor predictor
                               :data data-sorted
                               :vocab bert-vocab
                               :seq-length seq-length
                               :sentence (get data-sorted-map "20688153")}))

  (-> rlts (count))

  ;
  )

#_(def bert-base (m/load-checkpoint {:prefix (str bert-dir bert-base-prefix) :epoch 0}))

#_(-> (viz/plot-network
       (m/symbol bert-base)
       {"data0" [1 128]
        "data1" [1 128]
        "data2" [1]}
       {:title "bert"
        :node-attrs {:shape "oval" :fixedsize "false"}})
      (viz/render "bert" bert-dir))


#_(def bert-exported (m/load-checkpoint {:prefix (str bert-exported-dir "regression")
                                         :epoch 0}))

#_(-> (viz/plot-network
     (m/symbol bert-exported)
     {"data0" [1 512]
      "data1" [1 512]
      "data2" [1]}
     {:title "regression"
      :node-attrs {:shape "oval" :fixedsize "false"}})
    (viz/render "regression" bert-exported-dir))
