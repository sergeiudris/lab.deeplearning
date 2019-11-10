(ns tf.tensorflow
  (:require [clojure.repl :as repl])
  (:import
   (org.tensorflow Graph Session Tensor TensorFlow)))


#_(let [v (.getBytes "hello :)" "UTF-8")
        t (Tensor/create v)
        g (Graph.)]
    (do (-> g
            (.opBuilder "Const" "MyConst")
            (.setAttr "dtype" (.dataType t))
            (.setAttr "value" t)
            (.build)))
    (let [sess (Session. g)
          tensor-out (-> sess
                         (.runner)
                         (.fetch "MyConst")
                         (.run)
                         (.get 0))]
      (String. (.bytesValue tensor-out) "UTF-8")))


