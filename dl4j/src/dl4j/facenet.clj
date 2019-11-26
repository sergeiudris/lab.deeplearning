(ns dl4j.facenet
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [clojure.data.json :as json]
            [pad.prn.core :refer [linst linst-methods]])
  (:import
   org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
   org.deeplearning4j.nn.api.OptimizationAlgorithm
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$ListBuilder
   org.deeplearning4j.nn.conf.inputs.InputType
   org.deeplearning4j.nn.conf.layers.RnnOutputLayer$Builder
   org.deeplearning4j.nn.conf.layers.GravesLSTM$Builder
   org.deeplearning4j.nn.conf.layers.DenseLayer$Builder
   org.deeplearning4j.nn.conf.layers.OutputLayer$Builder
   org.deeplearning4j.nn.weights.WeightInit
   org.deeplearning4j.nn.conf.graph.MergeVertex
   org.deeplearning4j.nn.conf.graph.L2NormalizeVertex
   org.deeplearning4j.nn.transferlearning.TransferLearning
   org.deeplearning4j.optimize.listeners.ScoreIterationListener
   org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
   org.deeplearning4j.eval.Evaluation
   org.nd4j.linalg.learning.config.Nesterovs
   org.nd4j.linalg.learning.config.Adam
   org.nd4j.linalg.learning.config.Sgd
   org.nd4j.linalg.learning.config.AdaGrad
   org.nd4j.linalg.activations.Activation
   org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction
   org.deeplearning4j.zoo.model.helper.FaceNetHelper
   org.datavec.image.loader.LFWLoader)
  (:gen-class))

(comment
  
  
  ;
  )