(ns d2l.arxiv
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
            [pad.dataset.glove :refer [glove-filepath read-glove!]]
            [pad.dataset.arxiv :refer [categories>>data! data>>labeled
                                       data>>tokened tokened>>limited data>>embedded
                                       data>>padded fetch-arxiv-sample]]
            [pad.ml.nlp :refer [build-vocab]]
            [pad.dataset.bert :refer [read-vocab-json! fetch-bert-example fetch-bert-python] :as bert]
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

; using glove

; code/logic taken (or modified) from https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/cnn-text-classification
; all the credits go to the authors

(def categories ["cs" "econ" "eess" "math" "physics" "q-bio" "q-fin" "stat"])
(def categories ["cs" "physics" "math" "q-bio" ])
(def padding-token "</s>")
(def embedding-size 50)
(def num-filter 200)
(def dropout 0.5)

(def app-dir "/opt/app/")

(def opts
  {:glove.dir/shell app-dir
   :glove.dir/target (str app-dir "/tmp/data/glove/")
   :glove/embedding-size embedding-size

   :bert.dir/shell "/opt/app/"
   :bert.dir/from-mxnet-example (str app-dir "/tmp/data/bert/")
   :bert.dir/python-scripts "/opt/app/python/bert/"
   :bert.dir/mxnet "/root/.mxnet/"
   :bert.python/task "classification"
   :bert.python/seq-length 128
   :bert.python/prefix "bert-cls-4"
   :bert.python/num-classes 4
   :bert.python/output-dir (str app-dir "/tmp/data/bert-export/")
   
   :arxiv.dir/shell app-dir
   :arxiv.dir/target "/opt/app/tmp/data/arxiv/"
   :arxiv/categories categories})

#_(fetch-arxiv-sample opts)
#_(fetch-bert-python opts)


; from clojure mxnet example

(defn build-vocab-embeddings
  [vocab embeddings embedding-size]
  (->> (seq vocab)
       (map (fn [[word idx]]
              [word (or (get embeddings word)
                        (nd/->vec (random/uniform -0.25 0.25 [embedding-size])))]))
       (into {})))

#_(def vocab-embeddings (build-vocab-embeddings vocab glove embedding-size))
#_(count vocab-embeddings)
#_(first vocab-embeddings)

(defn get-data-symbol [num-embed sentence-size batch-size vocab-size pretrained-embedding]
  (if pretrained-embedding
    (sym/variable "data")
    (as-> (sym/variable "data") data
      (sym/embedding "vocab_embed" {:data data :input-dim vocab-size :output-dim num-embed})
      (sym/reshape {:data data :target-shape [batch-size 1 sentence-size num-embed]}))))


(defn make-filter-layers [{:keys [input-x num-embed sentence-size] :as config}
                          filter-size]
  (as-> (sym/convolution {:data input-x
                          :kernel [filter-size num-embed]
                          :num-filter num-filter}) data
    (sym/activation {:data data :act-type "relu"})
    (sym/pooling {:data data
                  :pool-type "max"
                  :kernel [(inc (- sentence-size filter-size)) 1]
                  :stride [1 1]})))

;;; convnet with multiple filter sizes
;; from Convolutional Neural Networks for Sentence Classification by Yoon Kim
(defn get-multi-filter-convnet [num-embed sentence-size batch-size 
                                vocab-size pretrained-embedding num-categories]
  (let [filter-list [3 4 5]
        input-x (get-data-symbol num-embed sentence-size batch-size vocab-size pretrained-embedding)
        polled-outputs (mapv #(make-filter-layers {:input-x input-x :num-embed num-embed :sentence-size sentence-size} %) filter-list)
        total-filters (* num-filter (count filter-list))
        concat (sym/concat "concat" nil polled-outputs {:dim 1})
        hpool (sym/reshape "hpool" {:data concat :target-shape [batch-size total-filters]})
        hdrop (if (pos? dropout) (sym/dropout "hdrop" {:data hpool :p dropout}) hpool)
        fc (sym/fully-connected  "fc1" {:data hdrop :num-hidden num-categories})]
    (sym/softmax-output "softmax" {:data fc})))

(defn data>>iters
  [{:keys [data batch-size embedding-size train-count valid-count dev]}]
  (let [data (shuffle data)

        data-x-train (->> data (take train-count) (map :embedded) (flatten) (vec))
        data-y-train (->> data (take train-count) (map :label) (vec))
        data-x-valid (->> data (drop train-count) (take valid-count) (map :embedded) (flatten) (vec))
        data-y-valid (->> data (drop train-count) (take valid-count) (map :label) (vec))
        sentence-size (->> data (first) :embedded (count))
        x-train  (nd/array data-x-train [train-count 1 sentence-size embedding-size] {:ctx dev})
        y-train  (nd/array data-y-train [train-count] {:ctx dev})
        
        x-valid  (nd/array data-x-valid [valid-count  1 sentence-size embedding-size] {:ctx dev})
        y-valid  (nd/array data-y-valid [valid-count] {:ctx dev})

        train-iter (mx-io/ndarray-iter [x-train]
                                       {:label [y-train]
                                        :label-name "softmax_label"
                                        :data-batch-size batch-size
                                        :last-batch-handle "pad"})

        valid-iter (mx-io/ndarray-iter [x-valid]
                                       {:label [y-valid]
                                        :label-name "softmax_label"
                                        :data-batch-size batch-size
                                        :last-batch-handle "pad"})]
    {:train-iter train-iter
     :valid-iter valid-iter
     :sentence-size sentence-size}))

(defn train
  [{:keys [batch-size vocab-size num-epoch iters num-categories contexts]}]
  (let [{:keys [train-iter valid-iter sentence-size]} iters]
    (prn "--starting training")
    (-> (get-multi-filter-convnet embedding-size sentence-size batch-size 
                                  vocab-size :glove num-categories)
        (m/module {:contexts contexts})
        (m/fit {:train-data train-iter
                :eval-data valid-iter
                :num-epoch num-epoch
                :fit-params (m/fit-params {:optimizer (optimizer/adam)})}))))

(comment

  (do
    (def glove (read-glove! (glove-filepath opts)))
    (def glove-vec (:vec glove))
    (def glove-to-embedding (:token-to-embedding glove))
    (def glove-to-token (:idx-to-token glove))
    (def glove-to-idx (:token-to-idx glove))

    (def data (categories>>data! opts))
    (def data-labeled (data>>labeled data))
    (def data-tokened (data>>tokened data-labeled))
    (def data-limited (tokened>>limited data-tokened))
    (def data-padded (data>>padded data-tokened))
    (def vocab (build-vocab (map :tokens data-padded)))
    (def vocab-embeddings (build-vocab-embeddings (:indexes vocab) glove embedding-size))
    (def data-embedded (data>>embedded data-padded vocab-embeddings))
    (def data-shuffled (shuffle data-embedded))
    )
  

  (count data)
  (->> data-shuffled (take 30) (map :setSpec))
  (->> data-shuffled (map :setSpec) (distinct))
  (->> data-shuffled (first) :embedded (count))
  (->> data-tokened (first) :tokens (count))
  (->> data-limited (first) :tokens (count))
  (->> data-shuffled (first) :tokens (count))

  (def batch-size 200)

  (def dev (context/gpu 0))

  (def iters (data>>iters {:data data-shuffled
                           :embedding-size embedding-size
                           :train-count 3200
                           :valid-count 800
                           :dev (context/cpu 0)
                           :batch-size batch-size}))

  (do
    (mx-io/reset (:train-iter iters))
    (mx-io/reset (:valid-iter iters)))

  (def mmod (train {:batch-size batch-size
                    :vocab-size (count vocab)
                    :num-epoch 10
                    :num-categories (count categories)
                    :iters iters
                    :contexts [dev]}))

  ;
  )

; using bert

; code/logic taken (or modified) from https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/bert
; all the credits go to the authors

(defn get-symbol-bert
  [pre-model num-classes dropout]
  (as-> (m/symbol pre-model) data
    (sym/dropout {:data data :p dropout})
    (sym/fully-connected "fc-finetune" {:data data :num-hidden num-classes})
    (sym/softmax-output "softmax" {:data data})))


(defn data>>bert-iter-data
  [data]
  (letfn [(data>>batch-column
           [data column-key]
           (->> data
                (mapv #(-> % :batch column-key))
                (flatten)
                (vec)))]
    (let [seq-length (->> data (first) :tokens (count))
          total (count data)]
      {:seq-length seq-length
       :data0 (data>>batch-column data :idxs)
       :data1 (data>>batch-column data :token-types)
       :data2 (data>>batch-column data :valid-length)
       :labels (->> data (map :label) (flatten) (vec))
       :total total
       :desc-data0 (mx-io/data-desc {:name "data0"
                                     :shape [total seq-length]
                                     :dtype dtype/FLOAT32
                                     :layout layout/NT})
       :desc-data1 (mx-io/data-desc {:name "data1"
                                     :shape [total seq-length]
                                     :dtype dtype/FLOAT32
                                     :layout layout/NT})
       :desc-data2 (mx-io/data-desc {:name "data2"
                                     :shape [total]
                                     :dtype dtype/FLOAT32
                                     :layout layout/N})
       :desc-label (mx-io/data-desc {:name "softmax_label"
                                     :shape [total]
                                     :dtype dtype/FLOAT32
                                     :layout layout/N})})))

(defn iter-data>>bert-iter
  [iter-data batch-size dev]
  (let [{:keys [data0 data1 data2
                labels total desc-data0 desc-data1
                desc-data2 desc-label seq-length]} iter-data]
    (mx-io/ndarray-iter {desc-data0 (nd/array data0 [total seq-length]
                                              {:ctx dev})
                         desc-data1 (nd/array data1 [total seq-length]
                                              {:ctx dev})
                         desc-data2 (nd/array data2 [total]
                                              {:ctx dev})}
                        {:label
                         {desc-label (nd/array labels [total]
                                               {:ctx dev})}
                         :data-batch-size batch-size})))



(defn train-bert!
  [{:keys [data dev num-epoch num-classes
           dropout batch-size
           train-count valid-count train-iter valid-iter]}]
  (let [bert-base (m/load-checkpoint {:prefix (str (:bert.python/output-dir opts)
                                                   (:bert.python/prefix opts))
                                      :epoch 0})
        model-sym (get-symbol-bert bert-base num-classes dropout)]
    (prn "--starting train")
    (as-> nil mmod
      (m/module model-sym {:contexts [dev]
                           :data-names ["data0" "data1" "data2"]})
      (m/fit mmod {:train-data train-iter
                   :eval-data valid-iter
                   :num-epoch num-epoch
                   :fit-params
                   (m/fit-params {:allow-missing true
                                  :arg-params (m/arg-params bert-base)
                                  :aux-params (m/aux-params bert-base)
                                  :optimizer (optimizer/adam {:learning-rate 5e-6 :epsilon 1e-9})
                                  :batch-end-callback (callback/speedometer batch-size 1)})})
      #_(m/save-checkpoint mmod {:prefix (str (:bert.python/output-dir opts)
                                              "bert-cls-4-tuned") :epoch num-epoch})
      mmod)))

(comment

  (do
    (def vocab (bert/read-vocab-json! (str (opts :bert.python/output-dir) "vocab.json")))
    (def data (categories>>data! opts))
    (def data-labeled (data>>labeled data))
    (def bert-tokened (bert/data>>tokened data-labeled :description))
    (def bert-padded (bert/data>>padded bert-tokened vocab {:seq-length 128} ))
    (def bert-shuffled (shuffle bert-padded))
    )

  (count data)
  (-> (get vocab "token_to_idx") (count))
  (-> (get vocab "idx_to_token") (count))

  (count bert-shuffled)
  (-> data-labeled (first))
  (-> bert-shuffled (first) :tokens (count))
  (-> bert-shuffled (first) :batch :token-types (count))
  (->> bert-shuffled (mapv #(count (:tokens %))) (apply max))
  (->> bert-shuffled (mapv #(-> % :batch :token-types (count))) (apply max))
  (def iter-data (data>>bert-iter-data bert-shuffled))
  (-> iter-data :data2 (count))
  (->> iter-data :labels (count))


  (def train-count 3200)
  (def valid-count 800)
  (def batch-size 32)

  (->> bert-shuffled (take train-count) (map :idxs) (flatten) (count)) ; 1600
  (->> bert-shuffled (take train-count) (map :token-types) (flatten) (count)) ; 1600


  (def train-iter (iter-data>>bert-iter
                   (->> bert-shuffled (take train-count) (data>>bert-iter-data))
                   batch-size
                   (context/cpu)))

  (def valid-iter (iter-data>>bert-iter
                   (->> bert-shuffled (drop train-count) (take valid-count) (data>>bert-iter-data))
                   batch-size
                   (context/cpu)))

  (def mmod (train-bert! {:data bert-shuffled
                          :train-iter train-iter
                          :valid-iter valid-iter
                          :dev (context/cpu 0)
                          :num-classes (count categories)
                          :dropout 0.1
                          :batch-size batch-size
                          :num-epoch 3}))
  ; mem 10G with seq-length 128   


  ;
  )

