(ns d2l.house
  (:require [clojure.reflect :refer [reflect]]
            [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [pad.coll.core :refer [contained?]]
            [pad.io.core :refer [read-nth-line count-lines]]
            [pad.dataset.house-prices :refer [fetch-dataset
                                              csv-file>>edn-file!
                                              edn-file>>data!
                                              data-dir standardize]]
            [pad.core :refer [str-float? str>>float resolve-var]]
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

(def opts
  {:dir/shell "/opt/app/"
   :dir/target "/opt/app/tmp/data/house-prices/"})

#_(fetch-dataset opts)

#_(csv-file>>edn-file! {:filename (str (data-dir opts) "train.csv")
                        :filename-out (str (data-dir opts) "train.csv.txt")
                        :nulls ["NA"]
                        :row>>row-vals (fn [row]
                                         (-> row (rest) (butlast)))
                        :row>>score (fn [row]
                                      [(str>>float (last row))])})

#_(csv-file>>edn-file! {:filename (str (data-dir opts) "test.csv")
                        :filename-out (str (data-dir opts) "test.csv.txt")
                        :nulls ["NA"]
                        :row>>row-vals (fn [row]
                                         (-> row (rest)))
                        :row>>score (fn [row]
                                      [100000])})

(defn data>>XY
  [data]
  (let [n-samples (count data)
        n-features (-> data (first) :features (count))]
    {:X (nd/array (->> data (map :features) (flatten) (vec))
                  [n-samples n-features])
     
     ; <!!!> (standardize) is improvisation, need to be removed
     ; the right approach: implement log-rmse and use adam optimizer
     :Y (nd/array (->> data (map :score) (flatten) (standardize) (vec))
                  [n-samples 1])}))

(defn train-XY
  []
  (->>
   (str (data-dir opts) "train.csv.txt")
   (edn-file>>data!)
   (take 1200)
   (data>>XY)))

(defn test-XY
  []
  (->>
   (str (data-dir opts) "train.csv.txt")
   (edn-file>>data!)
   (drop 1200)
   (take 260)
   (data>>XY)))

#_(def train-xy (train-XY))
#_(def test-xy (test-XY))
#_(def labels (->> test-xy (map :score) (flatten)))
#_(standardize labels)

(def batch-size 200) ;; the batch size
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
  [{:keys [mdl]}]
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
     mdl
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

                ; use optimizer/adam instead
                :optimizer (optimizer/sgd
                            {:learning-rate 0.01
                             :momentum 0.001
                             :lr-scheduler (lr-scheduler/factor-scheduler 3000 0.9)})

                ; need log-rmse metric instead
                :eval-metric (eval-metric/rmse)})}))))


(def data-names ["data"])
(def label-names ["linear_regression_label"])

(comment


  (def model
    (m/module (get-symbol)
              {:data-names data-names
               :label-names label-names
               :contexts [(context/cpu)]}))


  (train {:mdl model})

  ;
  )


