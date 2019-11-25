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
    Sgd)
   (org.nd4j.linalg.activations
    Activation)
   (org.nd4j.linalg.lossfunctions
    LossFunctions LossFunctions$LossFunction)
   (org.nd4j.linalg.learning.regularization 
    Regularization L1Regularization L2Regularization))
  (:gen-class))


;; MultiLayerNetwork And ComputationGraph

(comment

  ;; Building a MultiLayerConfiguration
  
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

  (def multi-layer-network (MultiLayerNetwork. multi-layer-conf))
  ;
  )

(comment

  ;; Building a ComputationGraphConfiguration

  (def computation-graph-conf (-> (NeuralNetConfiguration$Builder.)
                                  (.seed 123)
                                  #_(.learningRate 0.1)
                                  #_(.iterations 1)
                                  (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                                  (.graphBuilder)
                                  (.addInputs (into-array ["input"]))
                                  (.addLayer "L1" (-> (DenseLayer$Builder.)
                                                      (.nIn 3)
                                                      (.nOut 4)
                                                      (.build)) (into-array ["input"]))
                                  (.addLayer "out1" (-> (OutputLayer$Builder.)
                                                        (.lossFunction LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
                                                        (.nIn 4)
                                                        (.nOut 3)
                                                        (.build)) (into-array ["L1"]))
                                  (.addLayer "out2" (-> (OutputLayer$Builder.)
                                                        (.lossFunction LossFunctions$LossFunction/MSE)
                                                        (.nIn 4)
                                                        (.nOut 2)
                                                        (.build)) (into-array ["L1"]))
                                  (.setOutputs (into-array ["out1"]))
                                  #_(.pretrain false)
                                  #_(.backdrop true)
                                  (.allowDisconnected true)
                                  (.build)))
  (type computation-graph-conf)
  (json/read-str (.toJson computation-graph-conf) :key-fn keyword)

  (def computation-graph (ComputationGraph. computation-graph-conf))

  ;
  )

(comment

  ;; More MultiLayerConfiguration Examples

  ;; Regularization
  ; You can add regularization in the higher level configuration in the network through 
  ; first allowing regularization through 'regularization(true)' and then chaining it to
  ;  a regularization algorithm -> 'l1()', l2()' etc as shown below:
  (def x (-> (NeuralNetConfiguration$Builder.)
             (.l2 1E-4)
             #_(.regularization (java.util.Arrays/asList (object-array [(L2Regularization. 1E-4)])))
             (.build)))
  (json/read-str (.toJson x))
  
  ;; Dropout connects
  ; When creating layers, you can add a dropout connection 
  ; by using 'dropout(<dropOut_factor>)'

  (def x (-> (NeuralNetConfiguration$Builder.)
             (.list)
             (.layer 0 (-> (DenseLayer$Builder.)
                           (.dropOut 0.8)
                           (.build)))
             (.build)))
  (json/read-str (.toJson x))

  ;; Bias initialization
  ; You can initialize the bias of a particular layer by using 'biasInit(<init_value>)'

  (def x (-> (NeuralNetConfiguration$Builder.)
             (.list)
             (.layer 0 (-> (DenseLayer$Builder.)
                           (.biasInit 0)
                           (.build)))
             (.build)))
  (json/read-str (.toJson x))


  ;; More ComputationGraphConfiguration Examples

  ;; Recurrent Network 
  ; with Skip Connections

  (def x (-> (NeuralNetConfiguration$Builder.)
             #_(.learningRate 0.01)
             (.graphBuilder)
             (.addInputs (into-array ["input"]))
             (.addLayer "L1" (-> (GravesLSTM$Builder.)
                                 (.nIn 5)
                                 (.nOut 5)
                                 (.build)) (into-array ["input"]))
             (.addLayer "L2" (-> (RnnOutputLayer$Builder.)
                                 (.nIn (+ 5 5))
                                 (.nOut 5)
                                 (.build)) (into-array ["input" "L1"]))
             (.setOutputs (into-array ["L2"]))
             (.build)))
  (json/read-str (.toJson x))

  ;; Multiple Inputs and Merge Vertex
  ; Here MergeVertex concatenates the layer outputs

  (def x (-> (NeuralNetConfiguration$Builder.)
             #_(.learningRate 0.01)
             (.graphBuilder)
             (.addInputs (into-array ["input1" "input2"]))
             (.addLayer "L1" (-> (DenseLayer$Builder.)
                                 (.nIn 3)
                                 (.nOut 4)
                                 (.build)) (into-array ["input1"]))
             (.addLayer "L2" (-> (DenseLayer$Builder.)
                                 (.nIn 3)
                                 (.nOut 4)
                                 (.build)) (into-array ["input2"]))
             (.addVertex "merge" (MergeVertex.) (into-array ["L1" "L2"]))
             (.addLayer "out" (-> (OutputLayer$Builder.)
                                  (.nIn (+ 4 4))
                                  (.nOut 3)
                                  (.build)) (into-array ["merge"]))
             (.setOutputs (into-array ["out"]))
             (.build)))
  (json/read-str (.toJson x))

  ;; Multi-Task Learning

  (def x (-> (NeuralNetConfiguration$Builder.)
             #_(.learningRate 0.01)
             (.graphBuilder)
             (.addInputs (into-array ["input"]))
             (.addLayer "L1" (-> (DenseLayer$Builder.)
                                 (.nIn 3)
                                 (.nOut 4)
                                 (.build)) (into-array ["input"]))
             (.addLayer "out1" (-> (OutputLayer$Builder.)
                                   (.lossFunction LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
                                   (.nIn 4)
                                   (.nOut 3)
                                   (.build)) (into-array ["L1"]))
             (.addLayer "out2" (-> (OutputLayer$Builder.)
                                   (.lossFunction LossFunctions$LossFunction/MSE)
                                   (.nIn 4)
                                   (.nOut 2)
                                   (.build)) (into-array ["L1"]))
             (.setOutputs (into-array ["out1" "out2"]))
             (.build)))
  (json/read-str (.toJson x))


  ;
  )


(comment

  ;; Logistic Regression

  (def conf (-> (NeuralNetConfiguration$Builder.)
                (.seed 123)
                (.updater (Sgd. 0.1))
                #_(.iterations 1)
                (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                (.list)
                (.layer 0 (-> (OutputLayer$Builder.)
                              (.nIn 784)
                              (.nOut 10)
                              (.weightInit WeightInit/XAVIER)
                              (.activation Activation/SOFTMAX)
                              (.build)))
                #_(.pretrain false)
                #_(.backprop true)
                (.build)))
  (json/read-str (.toJson conf))



  ;
  )