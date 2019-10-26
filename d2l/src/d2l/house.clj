(ns d2l.house
  (:require [clojure.reflect :refer [reflect]]
            [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [clojure.data.csv :refer [read-csv]]
            [pad.coll.core :refer [contained?]]
            [pad.io.core :refer [read-nth-line count-lines]]
            [pad.core :refer [str-float? str>>float resolve-var]]
            [pad.math.core :refer [vec-standard-deviation-2
                                   scalar-subtract elwise-divide
                                   vec-mean scalar-divide
                                   mk-one-hot-vec]]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.kvstore :as kvstore]
            [org.apache.clojure-mxnet.kvstore-server :as kvstore-server]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
            [org.apache.clojure-mxnet.resource-scope :as resource-scope]
            [org.apache.clojure-mxnet.ndarray :as nd]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as shape])
  (:gen-class))

(def data-dir "./tmp/data/house/")
(def model-prefix "tmp/model/house/test")

(defn load-data!
  []
  (when-not  (.exists (io/file (str data-dir "train.csv")))
    (do
      (:exit (sh "bash" "-c" "bash bin/data.sh house" :dir "/opt/app"))
      (sh "bash" "-c" "mkdir -p tmp/data/house" :dir "/opt/app")
      (sh "bash" "-c" "mkdir -p tmp/model/house" :dir "/opt/app")
      )))

#_(load-data!)

(defn read-column
  [filename idx]
  (with-open [reader (io/reader filename)]
    (let [data (read-csv reader)
          rows (rest data)]
      (mapv #(nth % idx) rows))))

#_(distinct (read-column (str data-dir "train.csv")  65))

(defn read-column-mdata
  [{:keys [nulls] :or {nulls []}}]
  (letfn [(val>>column-type [v]
            (cond
              (str-float? v) :float
              :else :string))
          (column-type [rows column-idx]
            (->>
             rows
             (map #(nth % column-idx))
             (take-while #(not (contained? % nulls)))
             #_(take-last 1)
             (last)
             (val>>column-type)))
          (column [a k v rows]
            (let [dtype (column-type rows k)]
              (assoc a k (merge
                          {:idx k
                           :val v
                           :dtype dtype}
                          (when (= dtype :string)
                            {:distinct (distinct (mapv #(nth % k) rows))})))))]
    (with-open [reader-train (io/reader (str data-dir "train.csv"))
                reader-test (io/reader (str data-dir "test.csv"))]
      (let [data-train (read-csv reader-train)
            data-test (read-csv reader-test)
            rows-train (map #(-> % (rest) (butlast)) data-train)
            rows-test (map #(-> % (rest)) data-test)
            attrs (-> (first data-train) (rest) (butlast))
            rows (concat rows-train rows-test)]
        (reduce-kv (fn [a k v]
                     (column a k v rows))
                   (sorted-map) (vec attrs))))
    ))

#_(with-open [reader (io/reader (str data-dir "train.csv"))]
    (let [data (read-csv reader)
          attrs (first data)
          rows (rest data)]
      (column-mdata attrs rows)))

#_(read-column-mdata {:nulls ["NA"]})

#_(def nulls ["NA"])
#_(contained? "NA" nulls)


(defn standardize
  [v]
  (scalar-divide (vec-standard-deviation-2 v) (scalar-subtract  (vec-mean v) v)))

#_(standardize [10 20 30 40])
#_(vec-mean [1 2 3])
#_(standardize [1 2 3 2])


(defn read-features
  [{:keys [filename nulls data>>rows data>>attrs]}]
  (letfn [(val-null-float? [v dtype]
                           (and  (= dtype :float) (contained? v nulls)))
          (val-string? [v dtype]
                       (= dtype :string))
          (val-float? [v dtype]
                      (= dtype :float))
          (col-idx-float? [idx colsm]
                          (= (get-in colsm [idx :dtype]) :float))
          (row>>row-nums [row colsm]
                         (map-indexed (fn [i x]
                                        (if  (col-idx-float? i colsm)
                                          (str>>float x)
                                          x)) row))
          (string-field>>val [colm v]
                             (->> (:distinct colm)
                                  (keep-indexed (fn [i x]
                                                  (if (= x v)
                                                    (mk-one-hot-vec (count (:distinct colm)) i)
                                                    nil)))
                                  (first)))
          (float-field>>val [colm v row row-mean])
          (row>>mean [row]
                     (->> row
                          (keep-indexed (fn [i v]
                                          (if (number? v) v nil)))
                          (vec-mean)))
          (row>>float-features [row]
                               (->> row
                                    (filter number?)
                                    (standardize)))
          (row>>string-features [row]
                                (filter coll? row))
          (attr>>val [idx v row rows colsm row-mean]
                     (let [colm (get colsm idx)
                           dtype (:dtype colm)]
                       (cond
                         (val-string? v dtype) (string-field>>val colm v)
                         (val-null-float? v dtype) row-mean
                         (val-float? v dtype) v
                         :else (throw (Exception. "uknown/missing column :dtype")))))]
    (with-open [reader (io/reader filename)]
      (let [data (read-csv reader)
            rows (data>>rows data)
            attrs (data>>attrs data)
            colsm (read-column-mdata {:nulls nulls})]
        (mapv (fn [row]
                (let [row-nums (row>>row-nums row colsm)
                      row-mean (row>>mean row-nums)
                      row-denulled (map-indexed
                                    (fn [idx v]
                                      (attr>>val idx v row rows colsm row-mean))
                                    row-nums)
                      float-features (row>>float-features row-denulled)
                      string-features (row>>string-features row-denulled)]
                  (vec (concat  float-features string-features)))) rows)))))

#_(read-nth-line (str data-dir "train.csv") 704)
#_(read-nth-line (str data-dir "test.csv") 1)

(defn init-data!
  []
  (do
    (def fields
      (->
       (read-nth-line (str data-dir "train.csv") 1)
       (str/split  #",")))

    (def attrs (-> fields (rest) (vec)))

    (def train-samples
      (->>
       (with-open [reader (io/reader (str data-dir "train.csv"))]
         (->> (read-csv reader)
              (rest)
              (mapv #(-> % (rest) (butlast) (vec)))))))

    (def test-samples
      (->>
       (with-open [reader (io/reader (str data-dir "test.csv"))]
         (->> (read-csv reader)
              (rest)
              (mapv #(-> % (rest)  (vec)))))))


    (def samples (vec (concat train-samples test-samples)))

    (def train-labels-raw (with-open [reader (io/reader (str data-dir "train.csv"))]
                            (->> (read-csv reader)
                                 (rest)
                                 (mapv #(-> % (nth 80) (str>>float))))))

    (def train-features-raw (read-features {:filename (str data-dir "train.csv")
                                            :nulls ["NA"]
                                            :data>>rows (fn [data]
                                                          (->> data
                                                               (rest)
                                                               (map #(rest (butlast %)))))
                                            :data>>attrs (fn [data]
                                                           (->
                                                            (first data)
                                                            (rest)
                                                            (butlast)))}))

    (def test-features-raw (read-features {:filename (str data-dir "test.csv")
                                           :nulls ["NA"]
                                           :data>>rows (fn [data]
                                                         (->> data
                                                              (rest)
                                                              (map #(rest %))))
                                           :data>>attrs (fn [data]
                                                          (with-open [reader (io/reader (str data-dir "train.csv"))]
                                                            (->
                                                             (read-csv reader)
                                                             (first)
                                                             (rest)
                                                             (butlast)))
                                                          #_(->
                                                             (first data)
                                                             (rest)))}))

    ; (def train-features (->> train-features-raw (take 1000) (flatten) (vec)))
    ; (def train-labels (->> train-labels-raw (take 1000)  (vec)))
    (def eval-features (->> train-features-raw (drop 1000) (flatten) (vec)))
    (def eval-labels (->> train-labels-raw (drop 1000)  (vec)))
    (def test-features (->> test-features-raw (take 10) (flatten) (vec)))
    
    (def train-features (->> train-features-raw  (flatten) (vec)))
    (def train-labels (->> train-labels-raw   (vec)))

  ;
    ))

#_(init-data!)

#_(count fields) ; 81
#_(count attrs) ; 80
#_(count train-samples) ; 1460
#_(count test-samples) ; 1459
#_(count train-labels-raw) ; 1460
#_(take 5 train-labels)

#_(count (first train-samples))
#_(count (first test-samples))
#_(count train-features-raw)
#_(count test-features-raw)
#_(count (var-get (resolve 'train-features))) ; 304000
#_(count (resolve-var 'train-features))
#_(count eval-features) ; 139840
#_(count train-labels) ; 1000
#_(count eval-labels) ; 460
#_(count test-features)
#_(count train-features)

#_(nth train-features-raw 703)
#_(count (nth train-features-raw 703))
#_(count (nth test-features-raw 703))
#_(count (flatten (nth train-features-raw 703))) ; 354
#_(count (flatten (nth test-features-raw 703))) ; 354

#_(read-nth-line (str data-dir "train.csv") 9)
#_(nth train-features-raw 7) ; second value should be 0 (NA is normalized to 0)



(def batch-size 10) ;; the batch size
(def optimizer (optimizer/sgd {:learning-rate 0.01 :momentum 0.0}))
(def eval-metric (eval-metric/accuracy))
(def num-epoch 100) ;; the number of training epochs
(def kvstore "local") ;; the kvstore type
;;; Note to run distributed you might need to complile the engine with an option set
(def role "worker") ;; scheduler/server/worker
(def scheduler-host nil) ;; scheduler hostame/ ip address
(def scheduler-port 0) ;; scheduler port
(def num-workers 1) ;; # of workers
(def num-servers 1) ;; # of servers

(defn train
  [{:keys [contexts batch-size  train-features train-features-shape
           eval-features eval-features-shape
           train-labels train-labels-shape
           eval-labels eval-labels-shape
           ]}]
  (letfn [(get-symbol
            []
            (as-> (sym/variable "data") data
              
              ; (sym/fully-connected "fc1" {:data data :num-hidden 64})
              ; (sym/activation "relu1" {:data data :act-type "relu"})

              (sym/fully-connected "fc2" {:data data :num-hidden 128})
              (sym/activation "relu2" {:data data :act-type "relu"})

              (sym/fully-connected "fc3" {:data data :num-hidden 64})
              (sym/activation "relu3" {:data data :act-type "relu"})

              (sym/fully-connected "fc4" {:data data :num-hidden 1})
              (sym/softmax-output "softmax" {:data data})))

          (train-data
            []
            (mx-io/ndarray-iter [(nd/array train-features train-features-shape)]
                                {:label [(nd/array train-labels train-labels-shape)]
                                 :label-name "softmax_label"
                                 :data-batch-size batch-size
                                 :last-batch-handle "pad"}))

          (eval-data
            []
            (mx-io/ndarray-iter [(nd/array eval-features eval-features-shape)]
                                {:label [(nd/array eval-labels eval-labels-shape)]
                                 :label-name "softmax_label"
                                 :data-batch-size batch-size
                                 :last-batch-handle "pad"}))]
    (resource-scope/with-let [_mod (m/module (get-symbol) {:contexts contexts})]
      (let [mxm (m/fit _mod {:train-data (train-data)
                  ; :eval-data (eval-data)
                             :num-epoch num-epoch
                             :fit-params (m/fit-params {:kvstore kvstore
                                                        :optimizer optimizer
                                                        :eval-metric eval-metric})})]
        (m/save-checkpoint mxm {:prefix model-prefix :epoch num-epoch}))
      (println "Finish fit"))))

(def envs (cond-> {"DMLC_ROLE" role}
            scheduler-host (merge {"DMLC_PS_ROOT_URI" scheduler-host
                                   "DMLC_PS_ROOT_PORT" (str scheduler-port)
                                   "DMLC_NUM_WORKER" (str num-workers)
                                   "DMLC_NUM_SERVER" (str num-servers)})))
(defn start
  ([devices] (start devices num-epoch))
  ([devices _num-epoch]
   (when scheduler-host
     (println "Initing PS enviornments with " envs)
     (kvstore-server/init envs))

   (if (not= "worker" role)
     (do
       (println "Start KVStoreServer for scheduler and servers")
       (kvstore-server/start))
     (do
       (println "Starting Training of MNIST ....")
       (println "Running with context devices of" devices)
       (when-not (bound? #'train-features)
         (init-data!))
       (train {:contexts devices
               :train-features  (resolve-var 'train-features)
               :train-features-shape [1460 354]
               :train-labels (resolve-var 'train-labels)
               :train-labels-shape [1460 1]
              ;  :eval-features  (resolve-var 'eval-features)
              ;  :eval-features-shape [460 304]
              ;  :eval-labels (resolve-var 'eval-labels)
              ;  :eval-labels-shape [460 1]
               :batch-size batch-size})))))


#_(time (start [(context/cpu)]))




#_(def eval-data (mx-io/ndarray-iter [(nd/array test-features [10 354])]
                                     {:label
                                      [(nd/array (->> train-labels (take 10) (vec)) [10 1])]
                                      :label-name "softmax_label"
                                      :data-batch-size 1
                                      :last-batch-handle "pad"}))

#_(def mxmod
    (-> (m/load-checkpoint {:prefix model-prefix
                            :epoch 100
                            :load-optimizer-states false})
        (m/bind {:data-shapes (mx-io/provide-data eval-data)
                 :label-shapes (mx-io/provide-label eval-data)})
        (m/init-params)))

#_(mx-io/data-desc {:name "data0"
                    :shape [1 354]
                    :dtype dtype/FLOAT32
                    :layout layout/NT})

#_(def results
    (m/predict mxmod {:eval-data eval-data}))

#_(nd/->vec results)

