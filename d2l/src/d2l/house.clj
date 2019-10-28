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

(def data-dir "./tmp/data/house/")
(def model-prefix "tmp/model/house/test")

(defn load-data!
  []
  (when-not  (.exists (io/file (str data-dir "train.csv")))
    (do
      (:exit (sh "bash" "-c" "bash bin/data.sh house" :dir "/opt/app"))
      (:exit (sh "bash" "-c" "sudo chmod -R 777 tmp/" :dir "/opt/app"))
      (sh "bash" "-c" "mkdir -p tmp/data/house" :dir "/opt/app")
      (sh "bash" "-c" "mkdir -p tmp/model/house" :dir "/opt/app")
      )))

#_(load-data!)

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

#_(read-column-mdata {:nulls ["NA"]})


#_(std [1 2 -1 0 4])
#_(std [10, 12, 23, 23, 16, 23, 21, 16]) ;4.898979485566356 4.8989794855664
#_(standarddev [1 2 -1 0 4])

(defn standardize
  [v]
  (->> (scalar-subtract  (vec-mean v) v)
       (scalar-divide (std v))))

#_(def a-row (with-open [reader (io/reader (str data-dir "train.csv"))]
               (let [data (read-csv reader)
                     rows (rest data)]
                 (->> (nth rows 0)
                      (rest)
                      (butlast)
                      (map str>>float)
                      (filter number?)
                      (vec)))))

#_(standardize [1 0 0 0])
#_(vec-mean [1 0 0 0])
#_(standardize [1 0 0 0 1/4])

#_(vec-mean [1 2 3])
#_(standardize [1 2 3 2])

#_(standardize a-row)
#_(standardize-2 a-row)

#_(vec-standard-deviation-2 a-row)
#_(standarddev a-row)
#_(vec-mean a-row)
#_(mean a-row)
#_(scalar-subtract  (vec-mean a-row) a-row)


(defn csv>>data
  [{:keys [filename nulls row>>row-vals row>>score]}]
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
          (row>>float-null-features [row ]
                                    (->> row
                                         (filter string?)
                                         (map (fn [_] 0))
                                         ))
          (row>>string-features [row]
            (filter coll? row))
          (attr>>val [idx v row rows colsm row-mean]
            (let [colm (get colsm idx)
                  dtype (:dtype colm)]
              (cond
                (val-string? v dtype) (string-field>>val colm v)
                (val-null-float? v dtype) v
                (val-float? v dtype) v
                :else (throw (Exception. "uknown/missing column :dtype")))))]
    (with-open [reader (io/reader filename)]
      (let [data (read-csv reader)
            rows (rest data)
            colsm (read-column-mdata {:nulls nulls})]
        (mapv (fn [row]
                (let [row-vals (row>>row-vals row)
                      row-nums (row>>row-nums row-vals colsm)
                      row-mean (row>>mean row-nums)
                      row-hots (map-indexed
                                (fn [idx v]
                                  (attr>>val idx v row-vals rows colsm row-mean))
                                row-nums)
                      float-features
                      #_(->> (range 0 36) (mapv (fn [_] (rand))))
                      (row>>float-features row-hots)
                      float-null-features (row>>float-null-features row-hots)

                      string-features (row>>string-features row-hots)]
                  {:id (first row)
                   :features (-> (concat  float-features float-null-features string-features) (flatten) (vec))
                   :score (row>>score row)})) rows)))))

(defn csv-file>>edn-file!
  [opts]
  (->>
   (csv>>data opts)
   (str)
   (spit (str (:filename opts) ".txt"))))

(defn edn-file>>data!
  [filename]
  (-> filename
      (slurp)
      (read-string)))

(defn data>>XY
  [data]
  (let [n-samples (count data)
        n-features (-> data (first) :features (count))]
    {:X (nd/array (->> data (map :features) (flatten) (vec))
                  [n-samples n-features])
     :Y (nd/array (->> data (map :score) (flatten) (vec))
                  [n-samples 1])}))

(defn train-XY
  []
  (->
   (str data-dir "train.csv.txt")
   (edn-file>>data!)
   (data>>XY)))

(defn test-XY
  []
  (->
   (str data-dir "test.csv.txt")
   (edn-file>>data!)
   (data>>XY)))

#_(def train-xy (train-XY))
#_(def test-xy (test-XY))

#_(csv-file>>edn-file! {:filename (str data-dir "train.csv")
                        :nulls ["NA"]
                        :row>>row-vals (fn [row]
                                         (-> row (rest) (butlast)))
                        :row>>score (fn [row]
                                      [(str>>float (last row))])})

#_(csv-file>>edn-file! {:filename (str data-dir "test.csv")
                        :nulls ["NA"]
                        :row>>row-vals (fn [row]
                                         (-> row (rest)))
                        :row>>score (fn [row]
                                      [100000])})

#_(-> (slurp (str data-dir "test.csv.txt")) (read-string) (count) )
#_(-> (slurp (str data-dir "train.csv.txt")) (read-string) (count))
#_(-> (slurp (str data-dir "train.csv.txt")) (read-string) (first) :features (count))
#_(-> (slurp (str data-dir "test.csv.txt")) (read-string)  (first) :features (count))

#_(read-nth-line (str data-dir "train.csv") 704)
#_(read-nth-line (str data-dir "test.csv") 1)


(def batch-size 2000) ;; the batch size
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

(defn get-symbol
  []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 512})
    (sym/activation "act1" {:data data :act-type "relu"})
    (sym/dropout "drop1" {:data data :p 0.5})
    (sym/fully-connected "fc2" {:data data :num-hidden 128})
    (sym/activation "act2" {:data data :act-type "relu"})
    (sym/dropout "drop2" {:data data :p 0.5})
    (sym/fully-connected "fc3" {:data data :num-hidden 16})
    (sym/activation "act3" {:data data :act-type "relu"})
    (sym/fully-connected "fc4" {:data data :num-hidden 1})
    (sym/linear-regression-output "linear_regression" {:data data})))

(defn train
  [{:keys [model-mod]}]
  (let [train-xy (train-XY)
        test-xy (test-XY)
        train-iter (mx-io/ndarray-iter [(:X train-xy)]
                                       {:label [(:Y train-xy)]
                                        :label-name "linear_regression_label"
                                        :data-batch-size batch-size})
        test-iter (mx-io/ndarray-iter [(:X test-xy)]
                                      {:label [(:Y test-xy)]
                                       :label-name "linear_regression_label"
                                       :data-batch-size batch-size})]
    (->
     model-mod
     (m/bind {:data-shapes (mx-io/provide-data train-iter)
              :label-shapes (mx-io/provide-label test-iter)})
     (m/fit  {:train-data train-iter
              :eval-data test-iter
              :num-epoch num-epoch
              :fit-params
              (m/fit-params
               {; :kvstore kvstore
                :initializer (initializer/xavier)
                ; :batch-end-callback (callback/speedometer batch-size 100)
                :optimizer (optimizer/sgd
                            {:learning-rate 0.01
                             :momentum 0.001
                             :lr-scheduler (lr-scheduler/factor-scheduler 3000 0.9)})
                :eval-metric (eval-metric/mse)})}))))

(comment

  (def data-names ["data"])
  (def label-names ["linear_regression_label"])

  (def model-mod
    (m/module (get-symbol)
              {:data-names data-names
               :label-names label-names
               :contexts [(context/cpu)]}))


  (train {:model-mod model-mod})

  ;
  )



