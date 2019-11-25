(ns dl4j.emnist
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string])

  (:import
   (org.deeplearning4j.datasets.iterator AbstractDataSetIterator)
   (org.deeplearning4j.datasets.iterator.impl EmnistDataSetIterator EmnistDataSetIterator$Set)
   (org.deeplearning4j.nn.api Classifier)
   (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
   (org.deeplearning4j.nn.graph ComputationGraph)
   (org.deeplearning4j.nn.conf BackpropType)
   (org.deeplearning4j.nn.conf.inputs InputType)
   (org.deeplearning4j.nn.conf.layers AbstractLSTM)
   (org.deeplearning4j.nn.weights WeightInit)
   (org.deeplearning4j.optimize.listeners CheckpointListener)
   (org.deeplearning4j.datasets.datavec RecordReaderMultiDataSetIterator)
   (org.deeplearning4j.eval Evaluation)
   (org.nd4j.linalg.learning.config AdaDelta) ; for different updaters like Adam, Nesterovs, etc
   (org.nd4j.linalg.activations Activation) ;  defines different activation functions like RELU, SOFTMAX, etc.
   (org.nd4j.linalg.lossfunctions LossFunctions); mean squared error, multiclass cross entropy, etc.
   )
  (:gen-class))

(def batch-size 16)

(comment

  (def emnist-set EmnistDataSetIterator$Set/BALANCED)
  (def emnist-train (EmnistDataSetIterator. emnist-set batch-size true))
  (def emnist-test (EmnistDataSetIterator. emnist-set batch-size false))
  
  

  ;
  )