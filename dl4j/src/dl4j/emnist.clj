(ns dl4j.emnist
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [pad.prn.core :refer [linst linst-methods]])
  (:import
   org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
   org.deeplearning4j.datasets.iterator.AbstractDataSetIterator
   org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
   org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator$Set
   org.deeplearning4j.nn.api.OptimizationAlgorithm
   org.deeplearning4j.nn.multilayer.MultiLayerNetwork
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$ListBuilder
   org.deeplearning4j.nn.conf.BackpropType
   org.deeplearning4j.nn.conf.layers.DenseLayer$Builder
   org.deeplearning4j.nn.conf.layers.OutputLayer$Builder
   org.deeplearning4j.nn.weights.WeightInit
   org.deeplearning4j.optimize.listeners.ScoreIterationListener
   org.nd4j.linalg.learning.config.Adam
   org.nd4j.linalg.activations.Activation
   org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction)
  (:gen-class))


(comment

  (def batch-size 16)

  (def emnist-set EmnistDataSetIterator$Set/BALANCED)
  (def emnist-train (EmnistDataSetIterator. emnist-set batch-size true))
  (def emnist-test (EmnistDataSetIterator. emnist-set batch-size false))

  (def output-num (EmnistDataSetIterator/numLabels emnist-set))
  (def rng-seed 123) ; integer for reproducability of a random number generator
  (def num-rows 28)
  (def num-columns 28)

  ; as of 2019-11-25 docs at http://deeplearning4j.org/ are v1.0.0-beta4
  ; but data can be loaded only with v1.0.0-beta5, which has different api
  ; lookup api manually
  ; git clone https://github.com/eclipse/deeplearning4j
  ; get checkout deeplearning4j-1.0.0-beta5

  (def conf (-> (NeuralNetConfiguration$Builder.)
                (.seed rng-seed)
                (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                (.updater (Adam.))
                (.l2 1.0E-4)
                (.list)
                (.layer (-> (DenseLayer$Builder.)
                            (.nIn (* num-rows num-columns)) ;  Number of input datapoints
                            (.nOut 1000) ; Number of output datapoints
                            (.activation Activation/RELU)
                            (.weightInit WeightInit/XAVIER)
                            (.build)))
                (.layer (-> (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
                            (.nIn 1000) ;  
                            (.nOut output-num)
                            (.activation Activation/SOFTMAX)
                            (.weightInit WeightInit/XAVIER)
                            (.build)))
                (.build)))
  (.setBackpropType conf BackpropType/Standard) ; default
  (type conf)
  (linst conf)

  (def network (MultiLayerNetwork. conf))
  (def _ (.init network))
  (def each-iterations 5)
  (def _ (.addListeners network (into-array [(ScoreIterationListener. each-iterations)])))
  (def num-epochs 1)
  ; https://github.com/eclipse/deeplearning4j/blob/deeplearning4j-1.0.0-beta5/deeplearning4j/deeplearning4j-data/deeplearning4j-utility-iterators/src/main/java/org/deeplearning4j/datasets/iterator/MultipleEpochsIterator.java
  (linst network)
  (.fit network emnist-train num-epochs)
  (.getEpochCount network)
  (linst-methods network)
  (.score network)

  (def evaluation (.evaluate network emnist-test))
  (.accuracy evaluation)
  (.precision evaluation)
  (.recall evaluation)
  #_(.confusionString evaluation)
  
  )