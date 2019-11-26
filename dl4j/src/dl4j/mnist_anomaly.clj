(ns dl4j.mnist-anomaly
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [clojure.data.json :as json]
            [pad.prn.core :refer [linst linst-methods]]
            [clojure.reflect :as reflect])
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
   (org.apache.commons.lang3.tuple ImmutablePair Pair)
   (org.apache.commons.lang3.reflect
    FieldUtils))
  (:gen-class))


(comment

  ; configure

  (def conf (-> (NeuralNetConfiguration$Builder.)
                (.seed 12345)
                #_(.iterations 1)
                (.weightInit WeightInit/XAVIER)
                (.updater (AdaGrad. 0.05))
                (.activation Activation/RELU)
                (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                (.l2 0.0001)

                (.list)
                (.setInputType (InputType/feedForward 784))
                (.layer 0 (-> (DenseLayer$Builder.)
                              (.nIn 784)
                              (.nOut 250)
                              (.build)))
                (.layer 1 (-> (DenseLayer$Builder.)
                              (.nIn 250)
                              (.nOut 10)
                              (.build)))
                (.layer 2 (-> (DenseLayer$Builder.)
                              (.nIn 10)
                              (.nOut 250)
                              (.build)))
                (.layer 3 (-> (OutputLayer$Builder.)
                              (.nIn 250)
                              (.nOut 784)
                              (.lossFunction LossFunctions$LossFunction/MSE)
                              (.build)))
                (.build)))

  (json/read-str (.toJson conf))

  (def net (MultiLayerNetwork. conf))
  (do (.addListeners net (into-array [(ScoreIterationListener. 1)])))
  (.getListeners net)

  (do
    (def iter (MnistDataSetIterator. 100 50000 false))
    (def features-train (java.util.ArrayList.))
    (def features-test (java.util.ArrayList.))
    (def labels-test (java.util.ArrayList.))
    (def rnd (java.util.Random. 12345)))

  (do
    (while (.hasNext iter)
      (let [next (.next iter)
            split (.splitTestAndTrain next 80 rnd)
            _ (.add features-train (.. split (getTrain) (getFeatures)))
            ds-test (.getTest split)
            _ (.add features-test (.getFeatures ds-test))
            indexes (Nd4j/argMax (.getLabels ds-test) (int-array [1]))
            _ (.add labels-test indexes)])))

  (type (into-array [1]))
  (type (int-array [1]))

  (.size features-train)
  (.size features-test)
  (.size labels-test)
  (.totalOutcomes iter)

  (linst iter)
  (.-numExamples iter) ; fail, protected
  (FieldUtils/readField iter "numExamples" true)

  ; train

  (def n-epochs 1)

  (doseq [epoch (range 0 n-epochs)]
    (doseq [data (iterator-seq (.iterator features-train))]
      (.fit net data data))
    (prn (str "epoch " epoch " complete")))


  ; evaluate

  (def lists-by-digit (java.util.HashMap.))

  (doseq [i (range 0 10)]
    (.put lists-by-digit i (java.util.ArrayList.)))
  (count lists-by-digit)

  (.rows (.get features-test 1))
  (.getRow (.get features-test 1) 0)

  (def d (.. (.get labels-test 1) (getDouble  1) (intValue)))
  (linst d)

  (def pik (atom nil))
  (linst @pik)
  (do @pik)
  (.length @pik)
  (linst @pik)

  (doseq [i (range 0 (.size features-test))]
    (let [test-data (.get features-test i)
          labels (.get labels-test i)]
      (doseq [j (range 0 (.rows test-data))]
        (let [example (.getRow test-data j)
              digit (.. labels (getDouble  j) (intValue))
              dset (DataSet. example example)
              _ (.reshape dset 28 28)
              ; _ (reset! pik dset)
              score (.score net dset) ; Exception: Input that is not a matrix; expected matrix (rank 2), got rank 1 array with shape [784]
              digit-all-pairs (.get lists-by-digit digit)]
          (.add digit-all-pairs (ImmutablePair. score example))))))


  ; to be continued once docs are in sync with the latest version

  ;
  )