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

; using glove

; code/logic taken (or modified) from https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/cnn-text-classification
; all the credits go to the authors

(def data-dir "./tmp/data/arxiv/")
(def glove-dir "./tmp/data/glove/")


(def categories ["cs" "econ" "eess" "math" "physics" "q-bio" "q-fin" "stat"])
(def categories ["cs"  "physics" ])
(def padding-token "</s>")
(def embedding-size 50)

(defn load-glove!
  []
  (:exit (sh "bash" "-c" "bash bin/data.sh glove" :dir "/opt/app")))

#_(load-glove!)

(defn glove-path
  [embedding-size]
  (format (str glove-dir "glove.6B.%dd.txt") embedding-size))

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
       (take 1000)
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

#_(def data (categories>>data! categories))

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

#_(def glove (read-glove! (glove-path embedding-size)))
#_(count glove) ; 400000
#_(get glove "information")


(defn clean-str [s]
  (-> s
      (string/replace #"^A-Za-z0-9(),!?'`]" " ")
      
      (string/replace #"\." " . ")
      (string/replace #"\"" "")
      
      (string/replace #"'s" " 's")
      (string/replace #"'ve" " 've")
      (string/replace #"n't" " n't")
      (string/replace #"'re" " 're")
      (string/replace #"'d" " 'd")
      (string/replace #"'ll" " 'll")
      (string/replace #"," " , ")
      (string/replace #"!" " ! ")
      (string/replace #"\(" " ( ")
      (string/replace #"\)" " ) ")
      (string/replace #"\?" " ? ")
      (string/replace #" {2,}" " ")
      (string/trim)))

#_(string/replace "$\\sigma>0$" #"\$" "")

(defn data>>labels
  "Maps article metadata into {label-name normalized-value}"
  [data]
  (let [categories (->> data (map :setSpec) (distinct) (vec))
        size (dec (count categories))]
    (->> categories (reduce-kv #(assoc %1 %3 (/ (float %2) size)) {}))))

(defn  data>>labeled
  "Adds :label to data "
  [data]
  (let [labels (data>>labels data)]
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
  [data ]
  (mapv #(assoc % :tokens (->> % :description (text>>tokens))) data))

(defn tokened>>limited
  "Limits the length of token (sentence)"
  [data & {:keys [tokens-limit]}]
  (mapv (fn [v]
          (update v :tokens #(->> % (take tokens-limit) (vec)))) data))

(defn tokens>>padded
  [tokens padding-token max-seq-length]
  (let [diff (- max-seq-length (count tokens))]
    (into tokens  (repeat diff padding-token))))

(defn data>>padded
  [data]
  (let [max-seq-length (->> data (mapv #(count (:tokens %))) (apply max))]
    (mapv #(assoc % :tokens
                  (-> % :tokens (tokens>>padded padding-token max-seq-length)))
          data)))

#_(def data-tokened (data>>tokened data-labeled))
#_(nth data-tokened 1000)

#_(def data-padded (data>>padded data-tokened))
#_(nth data-padded 1000)

(defn build-vocab
  "Returns {word idx}"
  [tokens]
  (let [words (flatten tokens)
        freq (reduce (fn [a word]
                       (update-in a [word] (fnil inc 0))) {} words)
        freq-sorted (sort-by second > freq)
        words-sorted (map first freq-sorted)]
    (->>
     (map vector words-sorted (range 0 (count words-sorted)))
     (into {})
     )))

#_(def vocab (build-vocab (map :tokens data-padded)))
#_(count vocab) ; 50953
#_(nth (seq vocab) 1)

(defn build-vocab-embeddings
  [vocab embeddings embedding-size]
  (->> (seq vocab)
       (map (fn [[word idx]]
               [word (or (get embeddings word)
                         (nd/->vec (random/uniform -0.25 0.25 [embedding-size])))]))
       (into {})
       ))

#_(def vocab-embeddings (build-vocab-embeddings vocab glove embedding-size))
#_(count vocab-embeddings)
#_(first vocab-embeddings)

(defn data>>embedded
  "Adds :embedded [[..]]"
  [data embeddings]
  (mapv
   (fn [v]
     (assoc v :embedded (mapv #(embeddings %) (:tokens v)))) data))

#_(def data-embedded (data>>embedded data-padded vocab-embeddings))
#_(-> data-embedded (nth 5000) :embedded (flatten) (count)) ; 26700 (* 534 50)
#_(->> data-embedded (map #(-> %  :embedded (flatten) (count))) (reduce + 0)); 213600000
; = 213600000 (* 8000 26700)
#_(-> data-embedded   (first) (dissoc :embedded :tokens))

#_(def data-shuffled (shuffle data))
#_(type data-shuffled)


; from clojure mxnet example

(def num-filter 200)
(def dropout 0.5)

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
  [{:keys [data batch-size embedding-size train-count valid-count]}]
  (let [data (shuffle data)

        data-x-train (->> data (take train-count) (map :embedded) (flatten) (vec))
        data-y-train (->> data (take train-count) (map :label) (vec))
        data-x-valid (->> data (drop train-count) (take valid-count) (map :embedded) (flatten) (vec))
        data-y-valid (->> data (drop train-count) (take valid-count) (map :label) (vec))
        sentence-size (->> data (first) :embedded (count))
        x-train  (nd/array data-x-train [train-count 1 sentence-size embedding-size])
        y-train  (nd/array data-y-train [train-count])

        x-valid  (nd/array data-x-valid [valid-count  1 sentence-size embedding-size])
        y-valid  (nd/array data-y-valid [valid-count])

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
  [{:keys [batch-size vocab-size num-epoch iters num-categories]}]
  (let [{:keys [train-iter valid-iter sentence-size]} iters]
    (prn "--starting training")
    (-> (get-multi-filter-convnet embedding-size sentence-size batch-size 
                                  vocab-size :glove num-categories)
        (m/module {:contexts [(context/cpu 0) ]})
        (m/fit {:train-data train-iter
                :eval-data valid-iter
                :num-epoch num-epoch
                :fit-params (m/fit-params {:optimizer (optimizer/adam)})}))))

#_(do
    (def data (categories>>data! categories))
    (def glove (read-glove! (glove-path embedding-size)))
    (def data-labeled (data>>labeled data))
    (def data-tokened (data>>tokened data-labeled))
    (def data-limited (tokened>>limited data-tokened :tokens-limit 128))
    (def data-padded (data>>padded data-tokened))
    (def vocab (build-vocab (map :tokens data-padded)))
    (def vocab-embeddings (build-vocab-embeddings vocab glove embedding-size))
    (def data-embedded (data>>embedded data-padded vocab-embeddings))
    (def data-shuffled (shuffle data-embedded))
    )

#_(count data-shuffled)
#_(->> data-shuffled (take 30) (map :setSpec))
#_(->> data-shuffled (map :setSpec) (distinct))
#_(->> data-shuffled (first) :embedded (count))
#_(->> data-tokened (first) :tokens (count))
#_(->> data-limited (first) :tokens (count))


(def batch-size 200)

(comment

  (def iters (data>>iters {:data data-shuffled
                           :embedding-size embedding-size
                           :train-count 1600
                           :valid-count 400
                           :batch-size batch-size}))

  (do
    (mx-io/reset (:train-iter iters))
    (mx-io/reset (:valid-iter iters)))

  (def mmod (train {:batch-size batch-size
                    :vocab-size (count vocab)
                    :num-epoch 10
                    :num-categories (count categories)
                    :iters iters}))

  ;
  )

; using bert

; code/logic taken (or modified) from https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/bert
; all the credits go to the authors

(def bert-dir "./tmp/data/bert/")

(defn load-bert!
  []
  (:exit (sh "bash" "-c" "bash bin/data.sh bert" :dir "/opt/app")))

#_(load-bert!)

(defn read-bert-vocab!
  []
  (json/parse-stream (io/reader (str bert-dir  "vocab.json"))))


(defn break-out-punctuation [s str-match]
  (->> (string/split (str s "<punc>") (re-pattern (str "\\" str-match)))
       (map #(string/replace % "<punc>" str-match))))

(defn break-out-punctuations [s]
  (if-let [target-char (first (re-seq #"[.,?!]" s))]
    (break-out-punctuation s target-char)
    [s]))

(defn text>>bert-tokens [s]
  (->> (string/split s #"\s+")
       (mapcat break-out-punctuations)
       (into [])))

(defn pad [tokens pad-item num]
  (if (>= (count tokens) num)
    tokens
    (into tokens (repeat (- num (count tokens)) pad-item))))

(defn tokens>>idxs
  [vocab tokens]
  (let [token-to-idx (get  vocab "token_to_idx")
        idx-unk (get token-to-idx "[UNK]")]
    (mapv #(get token-to-idx % idx-unk) tokens)))

(defn idxs>>tokens
  [vocab idxs]
  (let [idx-to-token (get  vocab "idx_to_token")]
    (mapv #(get idx-to-token %) idxs)))

(defn data>>bert-tokened
  [data]
  (mapv #(assoc % :tokens (-> % :description (string/lower-case) (text>>bert-tokens))) data))

(defn data>>bert-padded
  [data vocab]
  (let [max-tokens-length (->> data (mapv #(count (:tokens %))) (apply max))
        seq-length (inc max-tokens-length)]
    (->> data
         (mapv (fn [v]
                 (let [tokens (:tokens v)
                       valid-length (count tokens)
                       token-types (pad [] 0 seq-length)
                       tokens (->> tokens (concat ["[CLS]"]) (vec))
                       tokens (pad tokens "[PAD]" seq-length)
                       idxs (tokens>>idxs vocab tokens)
                       ]
                   (merge v {:batch {:idxs idxs
                                     :token-types token-types
                                     :valid-length [valid-length] }
                             :tokens tokens
                             })
                   ))
               ))))

#_(def bert-vocab (read-bert-vocab!))
#_(def data (categories>>data! categories))
#_(def data-labeled (data>>labeled data))
#_(-> data-labeled (first))
#_(def bert-tokened (data>>bert-tokened data-labeled))
#_(-> bert-tokened (first))
#_(->> bert-tokened (mapv #(count (:tokens %))) (apply max))
#_(def bert-padded (data>>bert-padded bert-tokened bert-vocab))
#_(-> bert-padded (first))


