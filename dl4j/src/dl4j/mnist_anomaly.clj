(ns dl4j.mnist-anomaly
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
    EmnistDataSetIterator EmnistDataSetIterator$Set
    MnistDataSetIterator)
   (org.deeplearning4j.nn.api
    Classifier OptimizationAlgorithm)
   (org.deeplearning4j.nn.multilayer
    MultiLayerNetwork)
   (org.deeplearning4j.nn.graph
    ComputationGraph)

   (org.deeplearning4j.nn.conf
    Updater
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
    RnnOutputLayer RnnOutputLayer$Builder
    GravesLSTM GravesLSTM$Builder
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
    Adam
    Sgd
    AdaGrad)
   (org.nd4j.linalg.activations
    Activation)
   (org.nd4j.linalg.lossfunctions
    LossFunctions LossFunctions$LossFunction)
   (org.nd4j.linalg.learning.regularization
    Regularization L1Regularization L2Regularization)
   (org.nd4j.linalg.dataset.api.iterator DataSetIterator)
   (org.nd4j.linalg.api.ndarray INDArray)
   (org.nd4j.linalg.dataset DataSet)
   (org.nd4j.linalg.factory Nd4j)
   (javax.swing AbstractAction)
   (java.awt.image BufferedImage)
   (org.apache.commons.lang3.tuple Pair)
   (org.apache.commons.lang3.tuple ImmutablePair Pair))
  (:gen-class))


(comment
  
  
  ;
  )