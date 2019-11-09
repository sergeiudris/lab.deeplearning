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
            [pad.ml.bert :as bert]
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

(def data-dir "./tmp/data/recom/")
(def bert-dir "./tmp/data/bert/")

(defn load-data!
  []
  (:exit (sh "bash" "-c" "bash bin/data.sh cmu_movies" :dir "/opt/app")))

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
                           :countries :genres])))

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


#_(def mdata (read-metadata!))
#_(def summs (read-summaries!))
#_(def data (data>>joined mdata summs))
#_(def bert-vocab (bert/read-vocab! (str bert-dir "vocab.json")))
#_(def data-tokened (bert/data>>tokened data #(:summary %)))
#_(def data-padded (bert/data>>padded data bert-vocab))

#_(count data-tokened)
#_(->> data-tokened (map #(count (:tokens %))) (apply max))
#_(def data-sorted (->> data-tokened (sort-by :tokens #(> (count %1) (count %2)))))
#_(-> data-sorted (nth 5) :tokens (count))
#_(-> data-sorted (nth 5) :summary)
#_(-> data-sorted (nth 5) :name)
#_(-> data-sorted (last) :tokens (count))
#_(-> data-padded (first) :tokens (count))





