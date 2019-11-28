(ns dl4j.ui
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [pad.core :refer [safe-deref-future]])
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
   org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction

   java.io.File
   org.deeplearning4j.ui.api.UIServer
   org.deeplearning4j.api.storage.StatsStorage
   org.deeplearning4j.ui.stats.StatsListener
   org.deeplearning4j.ui.storage.FileStatsStorage
   org.deeplearning4j.examples.userInterface.util.UIExampleUtils)
  (:gen-class))


(comment

  (def net (UIExampleUtils/getMnistNetwork))

  (def train-iter (UIExampleUtils/getMnistData))

  (def ui-server (UIServer/getInstance))
  #_(.stop ui-server)

  (def stats-storage (FileStatsStorage.
                      (File. (System/getProperty "java.io.tmpdir") "ui-stats.dl4j1")))

  (def listener-frequency 1)

  (.setListeners net (into-array [(StatsListener. stats-storage listener-frequency)]))

  (.attach ui-server stats-storage)



  (def fu (future-call (fn []
                         (prn "--started training")
                         (.fit net train-iter)
                         (prn "--finished training"))))

  (prn 3)
  (safe-deref-future fu)
  (future-cancel fu)

  (.score net)






  ;
  )
  