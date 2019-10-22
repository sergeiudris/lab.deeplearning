(ns d2l.core
  (:require [clojure.reflect :refer [reflect]]
            [clojure.pprint :as pp]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.ndarray :as nd]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as shape]))

(defn linst
  [v]
  (->> v
       reflect
       :members
       (filter #(contains? (:flags %) :public))
       pp/print-table))

(comment

  (def x (nd/arange 0 12))
  (.shape x)
  (linst (.shape x))
  (.get (.shape x) 0)
  (.. x shape (get 0))
  (.size x)
  (def x1 (nd/reshape x [3 4]))
  (.. x1 shape (get 1))
  (nd/shape x)
  (nd/shape x1)
  (nd/size x)
  (nd/size x1)
  (nd/reshape x [4 -1])
  (nd/empty [3 4])

  (nd/zeros [2 3 4])

  (nd/ones [2 3 4])

  (nd/array (flatten [[2 1 4] 3 [1 2 3 4] [4 3 2 1]]) [3 4])
  
  (random/normal 0 1 [3 4])
  
  

  ;
  )

#_(ping)