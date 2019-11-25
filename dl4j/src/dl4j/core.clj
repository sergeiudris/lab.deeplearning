(ns dl4j.core
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [pad.core])

  (:import
   (dl4j.java Example)
   (org.nd4j.list NDArrayList)
   (org.nd4j.linalg.factory Nd4j)
   (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
   (org.deeplearning4j.examples.feedforward.classification MLPClassifierLinear)
   )
  (:gen-class))


(comment

  (Example/hello)
  (type (into-array [""]))
  (MLPClassifierLinear/main (into-array [""]))

  ;
  )