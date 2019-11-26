(ns dl4j.facenet
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [clojure.data.json :as json]
            [pad.prn.core :refer [linst linst-methods list-ctors]])
  (:import
   org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
   org.deeplearning4j.nn.api.OptimizationAlgorithm
   org.deeplearning4j.nn.conf.ConvolutionMode
   org.deeplearning4j.nn.conf.WorkspaceMode
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$ListBuilder
   org.deeplearning4j.nn.conf.inputs.InputType
   org.deeplearning4j.nn.conf.CacheMode
   org.deeplearning4j.nn.weights.WeightInit
   org.deeplearning4j.nn.conf.graph.MergeVertex
   org.deeplearning4j.nn.conf.layers.ConvolutionLayer$AlgoMode
   org.deeplearning4j.nn.conf.graph.L2NormalizeVertex
   org.deeplearning4j.nn.transferlearning.TransferLearning$GraphBuilder
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
   org.deeplearning4j.zoo.model.FaceNetNN4Small2
   org.datavec.image.loader.LFWLoader)
  (:gen-class))

(def batch-size 48)
(def num-examples LFWLoader/NUM_IMAGES)
(def output-num LFWLoader/NUM_LABELS)
(def split-train-test 1.0)
(def random-seed 123)
(def input-shape (int-array [3 96 96]))
(def iterations 1)
(def updater (Adam. 0.1 0.9 0.999 0.01))
(def transfer-function Activation/RELU)
(def cache-mode (CacheMode/NONE))
(def workspace-mode (WorkspaceMode/ENABLED))
(def cudnn-algo-mode (ConvolutionLayer$AlgoMode/PREFER_FASTEST) )
(def embedding-size 128)

(comment

  (list-ctors FaceNetNN4Small2)

  (->> FaceNetNN4Small2
       (.getDeclaredConstructors)
       (map #(.toGenericString %))
       (mapv #(string/split % #",")))

  (def zoo-model (FaceNetNN4Small2.
                  random-seed
                  input-shape
                  output-num
                  updater
                  transfer-function
                  cache-mode
                  workspace-mode
                  cudnn-algo-mode
                  embedding-size))

  (json/read-str (.. zoo-model (conf) (toJson)))

  (def net (.init zoo-model))

  (do (.setListeners net (into-array [(ScoreIterationListener. 1)])))
  (.getListeners net)

  (prn (.summary net))
  (.print System/out (.summary net))

  (def input-whc (int-array [(aget input-shape 2) (aget input-shape 1) (aget input-shape 0)]))

  (def iter (LFWDataSetIterator.
             batch-size
             num-examples
             input-whc
             output-num
             false
             true
             split-train-test
             (java.util.Random. random-seed)))

  (def num-epochs 1)

  (doseq [epoch (range 0 num-epochs)]
    (.fit  net iter)
    (prn (str "epoch " epoch " complete")))
  
  (.score net)

  (def shipped (-> (TransferLearning$GraphBuilder. net)
                   (.setFeatureExtractor (into-array ["embeddings"]))
                   (.removeVertexAndConnections "lossLayer")
                   (.setOutputs (into-array ["embeddings"]))
                   (.build)))
  
  (json/read-str (.. shipped conf toJson ))

  (def ds (.next iter))

  (def embedding (.feedForward shipped (.getFeatures ds) false))

  ;
  )