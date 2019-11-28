(ns dl4j.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [dl4j.linear]
            [dl4j.emnist]
            [dl4j.tutorial]
            [dl4j.mnist-anomaly]
            [dl4j.facenet]
            [dl4j.sea-temp]
            [dl4j.instacart]
            [dl4j.clouds]
            [dl4j.word2vec]
            [dl4j.doc2vec-cls]
            [dl4j.doc2vec-infer]
            [dl4j.doc2vec-text]
            [dl4j.ui]
            [dl4j.glove]
   ;
            )
  (:import org.nd4j.linalg.api.ndarray.INDArray)
  (:gen-class))

(defn -main  [& args]
  (prn INDArray)
  (start-nrepl-server :port 7788))