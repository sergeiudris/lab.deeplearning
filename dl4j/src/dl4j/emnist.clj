(ns dl4j.emnist
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [pad.prn.core :refer [linst linst-methods]])
  (:import
   (org.deeplearning4j.datasets.iterator
    MultipleEpochsIterator
    AbstractDataSetIterator)
   (org.deeplearning4j.datasets.iterator.impl
    EmnistDataSetIterator EmnistDataSetIterator$Set)
   (org.deeplearning4j.nn.api
    Classifier OptimizationAlgorithm)
   (org.deeplearning4j.nn.multilayer
    MultiLayerNetwork)
   (org.deeplearning4j.nn.graph
    ComputationGraph)
   (org.deeplearning4j.nn.conf
    NeuralNetConfiguration
    NeuralNetConfiguration$Builder
    NeuralNetConfiguration$ListBuilder
    BackpropType)
   (org.deeplearning4j.nn.conf.inputs
    InputType)
   (org.deeplearning4j.nn.conf.layers
    DenseLayer DenseLayer$Builder
    OutputLayer OutputLayer$Builder)
   (org.deeplearning4j.nn.weights
    WeightInit)
   (org.deeplearning4j.optimize.listeners
    ScoreIterationListener)
   (org.deeplearning4j.datasets.datavec
    RecordReaderMultiDataSetIterator)
   (org.deeplearning4j.eval 
    Evaluation)
   (org.nd4j.linalg.learning.config 
    Adam) ; for different updaters like Adam, Nesterovs, etc
   (org.nd4j.linalg.activations 
    Activation) ;  defines different activation functions like RELU, SOFTMAX, etc.
   (org.nd4j.linalg.lossfunctions
    LossFunctions LossFunctions$LossFunction); mean squared error, multiclass cross entropy, etc.
   )
  (:gen-class))

(def batch-size 16)

(comment



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
  ; https://github.com/eclipse/deeplearning4j/tree/deeplearning4j-1.0.0-beta5/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf  

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
  #_(.setBackpropType conf BackpropType/Standard) ; default
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
  (.score network) ; accuracy
  (.fit network emnist-train 5)


  ;
  )