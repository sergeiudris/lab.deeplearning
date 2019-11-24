(ns dl4j.core
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string])

  (:import
   (dl4j.java Example)
   (org.nd4j.list NDArrayList)
   #_(org.deeplearning4j.nn.multilayer MultiLayerNetwork))
  (:gen-class))

(comment

  (Example/hello)

  ;
  )