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

#_(ping)

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

  (nd/slice c 0)

  (nd/array [1] [1])

  (def A (nd/reshape (nd/arange 0 20) [4 5]))

  (nd/sum A)
  (nd/mean A)
  (nd// (nd/sum A) (nd/size A))

  (def x (nd/arange 0 4))
  (def A (nd/reshape (nd/arange 0 20) [5 4]))
  (def c (nd/dot A x))


  (def A (nd/reshape (nd/arange 0 20) [5 4]))
  (def B (nd/ones [4 3]))
  (def c (nd/dot A B))

  (def x (nd/arange 0 4))
  ; l2 norm
  (nd/norm x)
  ; l1 norm
  (nd/sum (nd/abs x))

  (def n 100000)
  (def a (nd/ones [n]))
  (def b (nd/ones [n]))
  (time (nd/+ a b))
  (nd/shape a)
  (nd/->vec (nd/shape a))
  (nd/size a)
  ;
  )

; linear regression from scratch

(defn synthetic-data
  "generates y = Xw + b + noise "
  [w b num-examples]
  (let [X (random/normal 0 1 [num-examples (nd/size w)])
        y0 (-> (nd/dot X w) (nd/+ b))
        noise (random/normal 0 0.01 (nd/->vec (nd/shape y0)))
        y (nd/+ y0 noise)]
    [X y]))

(comment
  (def true-w (nd/array [2 -3.4] [2]))
  (def true-b 4.2)
  
  (def data (synthetic-data true-w true-b 1000 ))

  ;
  )