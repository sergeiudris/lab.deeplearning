(ns dl4j.mnist-anomaly
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [clojure.data.json :as json]
            [pad.prn.core :refer [linst linst-methods]]
            [clojure.reflect :as reflect])
  (:import
   org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
   org.deeplearning4j.nn.api.OptimizationAlgorithm
   org.deeplearning4j.nn.multilayer.MultiLayerNetwork
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$ListBuilder
   org.deeplearning4j.nn.conf.layers.DenseLayer$Builder
   org.deeplearning4j.nn.conf.layers.OutputLayer$Builder
   org.deeplearning4j.nn.weights.WeightInit
   org.deeplearning4j.optimize.listeners.ScoreIterationListener
   org.nd4j.linalg.activations.Activation
   org.nd4j.linalg.learning.config.AdaGrad
   org.nd4j.linalg.factory.Nd4j
   org.deeplearning4j.nn.conf.inputs.InputType
   org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction)
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

  (json/read-str (.toJson conf) :key-fn keyword)

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

  (def n-epochs 10)
  
  (def fu (future-call (fn []
                         (prn "--started training")
                         (doseq [epoch (range 0 n-epochs)
                                 :while (not (.isInterrupted (Thread/currentThread)))]
                           (doseq [data (iterator-seq (.iterator features-train))]
                             (.fit net data data))
                           (prn (str "epoch " epoch " complete")))
                         (prn "--finished training"))))
  
  (future-cancel fu)
  (future-done? fu)
  
  (future-cancelled? fu)

  (Thread/interrupted)
  (Thread/currentThread)
  (linst (Thread/currentThread))
  (.getId (Thread/currentThread))
  (.isInterrupted (Thread/currentThread))


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