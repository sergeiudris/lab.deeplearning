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
            [org.apache.clojure-mxnet.shape :as shape])
  (:gen-class))

; all the credits to https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/cnn-text-classification

(def data-dir "./tmp/data/arxiv/")
(def glove-dir "./tmp/data/glove/")


(def categories ["cs" "econ" "eess" "math" "physics" "q-bio" "q-fin" "stat"])

(defn load-glove!
  []
  (:exit (sh "bash" "-c" "bash bin/data.sh glove" :dir "/opt/app")))

(defn glove-path
  [embedding-size]
  (format (str glove-dir "glove.6B.%dd.txt") embedding-size))

#_(load-glove!)

(defn axriv-xml-file>>article-vec!
  "Returns a vector of  articles' metadata in xml-edn"
  [path]
  (->> path
       (clojure.xml/parse)
       :content
       (last)
       :content
       (butlast)))

#_(def recrods (axriv-xml-file>>article-vec! (str data-dir "oai2-cs-1000.xml")) )
#_(count records)
#_(first records)

(defn arxiv-xml>>data
  [xml]
  {:identifier (-> xml :content (first) :content (first) :content (first))
   :title (-> xml :content (second) :content (first) :content (first) :content (first))
   :setSpec (-> xml :content (first) :content (last) :content (first))
   :description (->> xml :content (second) :content (first) :content
                     (reduce #(when (= (:tag %2) :dc:description) (reduced %2)))
                     :content (first))})

(defn arxiv-xml>>edn!
  "Reads xml, xforms and saves to edn"
  [filename]
  (->> filename
       (axriv-xml-file>>article-vec!)
       (map arxiv-xml>>data)
       ))

#_(def cs-data (vec (arxiv-xml>>edn! (str data-dir "oai2-cs-1000.xml"))))
#_(take 5 cs-data)
#_(count (take-while :description cs-data))

(defn xml-file>>edn-file!
  [in-file out-file]
  (as-> nil v
    (arxiv-xml>>edn! in-file)
    (vec v)
    (str v)
    #_(with-out-str (pp/pprint v))
    (spit out-file v)))

#_(xml-file>>edn-file! (str data-dir "oai2-cs-1000.xml")
                       (str data-dir "oai2-cs-1000.edn.txt"))

#_(doseq [c categories]
    (xml-file>>edn-file! (str data-dir "oai2-" c "-1000.xml")
                         (str data-dir "oai2-" c "-1000.edn.txt")))

#_(def data (->> categories
                 (mapcat (fn [c]
                           (->> (str data-dir "oai2-" c "-1000.xml")
                                (arxiv-xml>>edn!)
                                (vec))))
                 (vec)))

#_(count data)
#_(->> data (map :setSpec) (distinct))

(defn categories>>data!
  [categories]
  (->> categories
       (mapcat (fn [c]
                 (->> (str data-dir "oai2-" c "-1000.xml")
                      (arxiv-xml>>edn!)
                      (vec))))))

(defn lines>>word-embeddings
  "maps lines into  [[word embeddings]..]"
  [lines]
  (for [^String line lines
        :let [fields (.split line " ")]]
    [(first fields)
     (mapv #(Float/parseFloat %) (rest fields))]))

(defn read-glove!
  "Reads glove file into {word embeddings}"
  [path]
  (prn "-- reading glove from " path)
  (->> (io/reader path)
       (line-seq)
       (lines>>word-embeddings)
       (into {})))

#_(def glove (read-glove! (glove-path 50)))
#_(count glove) ; 400000
#_(get glove "information")


(defn clean-str [s]
  (-> s
      (string/replace #"^A-Za-z0-9(),!?'`]" " ")
      (string/replace #"'s" " 's")
      (string/replace #"'ve" " 've")
      (string/replace #"n't" " n't")
      (string/replace #"'re" " 're")
      (string/replace #"'d" " 'd")
      (string/replace #"'ll" " 'll")
      (string/replace #"," " , ")
      (string/replace #"\." " . ")
      (string/replace #"!" " ! ")
      (string/replace #"\(" " ( ")
      (string/replace #"\)" " ) ")
      (string/replace #"\?" " ? ")
      (string/replace #" {2,}" " ")
      (string/trim)))

(defn data>>labels
  "Maps article metadata into {label-name normalized-value}"
  [data]
  (let [categories (->> data (map :setSpec) (distinct) (vec))
        size (dec (count categories))]
    (->> categories (reduce-kv #(assoc %1 %3 (/ (float %2) size)) {}))))

(defn  data>>labeled
  "Adds :label to data "
  [data]
  (let [labels (arxiv-data>>labels data)]
    (mapv #(->> (get labels (:setSpec %))
                (assoc % :label)) data)))

#_(def data-labeled (data>>labeled data))
#_(data>>labels data)
#_(count data-labeled)
#_(nth data-labeled 7000)

(defn text>>tokens
  [text]
  (-> text
      (clean-str)
      (string/split #"\s+")))

(defn data>>tokened
  "Adds :tokens to datapoints"
  [data]
  (mapv #(assoc % :tokens (-> % :description (text>>tokens))) data))

#_(def data-tokened (data>>tokened data-labeled))
#_(nth data-tokened 1000)
