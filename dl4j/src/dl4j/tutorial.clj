(ns dl4j.tutorial
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [clojure.data.json :as json]
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
    MultiLayerConfiguration
    ComputationGraphConfiguration
    NeuralNetConfiguration
    NeuralNetConfiguration$Builder
    NeuralNetConfiguration$ListBuilder
    BackpropType)
   (org.deeplearning4j.nn.conf.inputs
    InputType)
   (org.deeplearning4j.nn.conf.graph
    MergeVertex)
   (org.deeplearning4j.nn.conf.layers
    RnnOutputLayer
    GravesLSTM
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
    Nesterovs
    Adam)
   (org.nd4j.linalg.activations
    Activation)
   (org.nd4j.linalg.lossfunctions
    LossFunctions LossFunctions$LossFunction))
  (:gen-class))


(comment
  ; MultiLayerNetwork And ComputationGraph

  (def multi-layer-conf (-> (NeuralNetConfiguration$Builder.)
                            (.seed 123)
                            #_(.learningRate 0.1)
                            #_(.iterations 1)
                            (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                            (.list)
                            (.layer 0 (-> (DenseLayer$Builder.)
                                          (.nIn 784)
                                          (.nOut 100)
                                          (.weightInit WeightInit/XAVIER)
                                          (.activation Activation/RELU)
                                          (.build)))
                            (.layer 1 (-> (OutputLayer$Builder.)
                                          (.nIn 100)
                                          (.nOut 10)
                                          (.weightInit WeightInit/XAVIER)
                                          (.activation Activation/SIGMOID)
                                          (.build)))
                            #_(.pretrain false)
                            #_(.backdrop true)
                            ; Invalid output layer configuration for layer "layer1": sigmoid activation function in combination with LossMCXENT 
                            (.validateOutputLayerConfig false)
                            (.build)))
  
  (type (.toJson multi-layer-conf))
  (json/read-str (.toJson multi-layer-conf) :key-fn keyword)





  ;
  )