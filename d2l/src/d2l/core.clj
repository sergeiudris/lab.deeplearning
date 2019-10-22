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

  (def x2 (nd/array [1 2 4 8] [4]))
  (def x3 (nd/* (nd/ones-like x2) 2))
  (nd/+ x2 x3)
  (nd/- x2 x3)
  (nd/* x2 x3)
  (nd// x2 x3)
  (nd/exp x2)
  
   ; inplace
  (nd/+= x2 x3)

  (def x (-> (nd/arange 0 12) (nd/reshape [3 4])))
  (def y (nd/array (flatten [[2 1 4 3] [1 2 3 4] [4 3 2 1]]) [3 4]))
  (nd/dot x (nd/transpose y))

  (nd/concatenate [x y] {:axis 0})
  (nd/concatenate [x y] {:axis 1})
  
  (nd/equal x y)
  (nd/< x y)
  (nd/>= x y)
  
  
  (nd/sum x)
  (nd/->double-vec (nd/norm (nd/sum x)))
  (nd/->raw (nd/sum x))
  (nd/->vec (nd/sum x))
  
  (nd/->vec (nd/norm x))
  (first (nd/->vec (nd/norm x)))
  
  (def a (-> (nd/arange 0 3) (nd/reshape [3 1])))
  (def b (-> (nd/arange 0 2) (nd/reshape [1 2])))
  
  (def c (nd/broadcast-add a b))
  
  (nd/slice c 0 )
  
  
 
  
  
  
  ;
  )

#_(ping)