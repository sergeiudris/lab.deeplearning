(ns tf.tensorflow
  (:require [clojure.repl :as repl])
  (:import
   (org.tensorflow Graph Session Tensor TensorFlow)))

(defn version
  []
  (TensorFlow/version))

(defn graph
  []
  (Graph.))

(defn tensor
  ([a]
   (Tensor/create a))
  ([a b]
   (Tensor/create a b))
  ([a b c]
   (Tensor/create a b c)))

#_(def g (graph))
#_(def t (tensor (.getBytes "hello" "UTF-8") ))

(defn op-builder
  [g a b]
  (.opBuilder g a b))

(defn set-attr
  [g a b]
  (.setAttr g a b))

(defn t>datatype
  [t]
  (.dataType t))

(defn build
  [g]
  (.build g))

(defn session
  [g]
  (Session. g))

#_(let [t (tensor (.getBytes "hello!" "UTF-8"))
        g (graph)]
    (do (-> g
            (op-builder "Const" "MyConst")
            (set-attr "dtype" (t>datatype t))
            (set-attr "value" t)
            (build)))
    (let [sess (session g)
          tensor-out (-> sess
                         (.runner)
                         (.fetch "MyConst")
                         (.run)
                         (.get 0))]
      (String. (.bytesValue tensor-out) "UTF-8")))


