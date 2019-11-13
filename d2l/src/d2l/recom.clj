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
            [pad.mxnet.bert :as bert]
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
            )
  (:gen-class))

(def data-dir "./tmp/data/recom/MovieSummaries/")
(def models-dir "./tmp/models/")
(def bert-model-filename "bert_12_768_12_book_corpus_wiki_en_uncased-75cc780f.params")
(def bert-vocab-filename "book_corpus_wiki_en_uncased-a6607397.vocab")

(def bert-dir "./tmp/data/bert/")
(def bert-base-prefix "static_bert_base_net")
(def bert-base-vocab-filename "vocab.json")

(def bert-exported-dir "./tmp/models/bert_exported/")


(defn load-data!
  []
  (:exit (sh "bash" "-c" "bash bin/data.sh recom" :dir "/opt/app"))
  (when-not (.exists (io/file (str models-dir bert-model-filename)))
    (:out (sh "bash" "-c" "bash bin/data.sh bert_base" :dir "/opt/app"))))

#_(load-data!)
#_(-> (sh "bash" "-c" (format "cat %s" (str data-dir "README.txt"))) :out)

(comment
  
  (read-nth-line (str data-dir "plot_summaries.txt") 1)
  (-> (read-nth-line (str data-dir "movie.metadata.tsv") 2) (string/split #"\t"))
  
  
  ;
  )

(defn csv-file>>vec!
  [filename & {:keys [separator] :or {separator "\t"}}]
  (with-open [reader (io/reader filename)]
    (->> reader
         (line-seq)
         (map #(string/split % (re-pattern (str separator))))
         (vec))))

#_(def mdata (csv-file>>vec! (str data-dir "movie.metadata.tsv")))
#_(-> mdata (first))

(defn csv-vec>>entities
  [columns data]
  (->> data
       (mapv #(reduce-kv
               (fn [a i v]
                 (assoc a (get columns i) v)) {} %))))

(defn read-metadata!
  []
  (->> (str data-dir "movie.metadata.tsv")
       (csv-file>>vec!)
       (csv-vec>>entities [:id-wiki :id-freebase :name
                           :release-date :box-office
                           :runtime :languages
                           :countries :genres])
       (mapv (fn [v]
               (update v :box-office #(if-not (empty? %) (Float/parseFloat %) 0))))
       ))

#_(def mdata (read-metadata! ))
#_(first mdata)

(defn read-summaries!
  []
  (->> (str data-dir "plot_summaries.txt")
       (csv-file>>vec!)
       (csv-vec>>entities [:id-wiki :summary])
       (reduce #(assoc %1 (:id-wiki %2) %2) {} )
       ))

#_(def summs (read-summaries!))
#_(first summs)
#_(count summs)

(defn data>>joined
  [mdata summs]
  (->> mdata
       (mapv (fn [v]
               (assoc v :summary (-> summs (get (:id-wiki v)) :summary))))
       (filterv :summary)))

#_(def data (data>>joined mdata summs))
#_(nth data 3)
#_(count data)
#_(->> data (filter :summary) (count))

#_(do
    (def mdata (read-metadata!))
    (def summs (read-summaries!))
    (def data (data>>joined mdata summs))
    (def bert-vocab (bert/read-vocab-json! (str bert-dir bert-base-vocab-filename)))
    (def data-tokened (bert/data>>tokened data #(:summary %)))
    (def data-filtered (->> data-tokened (filterv #(<= (-> % :tokens (count)) 510))))
    (def data-padded (bert/data>>padded data-filtered bert-vocab))
    (def data-sorted (->> data-padded (sort-by :box-office >)))
    (def seq-length (->> data-padded (first) :tokens (count) ))
    (def data-sorted-map (->> data-sorted (reduce #(assoc %1 (:id-wiki %2) %2) {})  ))
    )

#_(->> (get bert-vocab "idx_to_token") (count)) ; 30522
#_(->> (get bert-vocab "token_to_idx") (count)) ; 30522

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

(defn find-equivalent
  "Get the fine-tuned model's opinion on whether two sentences are equivalent:"
  [{:keys [predictor seq-length data target]}]
  (->> data
       (mapv (fn [v]
               (let [pair [target v]]
                 (->>
                  (infer/predict-with-ndarray
                   predictor
                   [(nd/array (bert/data>>batch-column pair :idxs) [1 seq-length])
                    (nd/array (bert/data>>batch-column pair :token-types) [1 seq-length])
                    (nd/array (bert/data>>batch-column pair :valid-length) [1])])
                  (first)
                  (nd/->vec)))))
       (sort-by first >)))

(comment

  (def fine-tuned-predictor
    (infer/create-predictor
     (infer/model-factory
      (str bert-dir bert-base-prefix)
      [{:name "data0" :shape [1 seq-length] :dtype dtype/FLOAT32 :layout layout/NT}
       {:name "data1" :shape [1 seq-length] :dtype dtype/FLOAT32 :layout layout/NT}
       {:name "data2" :shape [1]            :dtype dtype/FLOAT32 :layout layout/N}])
     {:epoch 0}))

  (def rlt (find-equivalent {:predictor fine-tuned-predictor
                             :data data-sorted
                             :seq-length seq-length
                             :target (get data-sorted-map "161190")}))
  
  
  
  )

(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym shape-map path]}]
  (let [dot (viz/plot-network
             model-sym
             shape-map
             {:title model-name
              :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot model-name path)))

#_(def bert-base (m/load-checkpoint {:prefix (str bert-dir bert-base-prefix) :epoch 0}))
#_(render-model! {:model-name "bert"
                  :model-sym (m/symbol bert-base)
                  :shape-map {"data0" [1 128]
                              "data1" [1 128]
                              "data2" [1]}
                  :path bert-dir})

#_(def bert-exported (m/load-checkpoint {:prefix (str bert-exported-dir "regression")
                                         :epoch 0}))
#_(render-model! {:model-name "regression"
                  :model-sym (m/symbol bert-exported)
                  :shape-map {"data0" [1 512]
                              "data1" [1 512]
                              "data2" [1]}
                  :path bert-exported-dir})