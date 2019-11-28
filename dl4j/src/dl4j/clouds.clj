(ns dl4j.clouds
  (:require [clojure.pprint :as pp]
            [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as string]
            [clojure.data.json :as json]
            [pad.prn.core :refer [linst linst-methods]])
  (:import
   org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
   org.deeplearning4j.datasets.iterator.AbstractDataSetIterator
   org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
   org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator$Set
   org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
   org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator$AlignmentMode
   org.deeplearning4j.nn.api.OptimizationAlgorithm
   org.deeplearning4j.nn.multilayer.MultiLayerNetwork
   org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator$Builder
   org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator$AlignmentMode
   org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$ListBuilder
   org.deeplearning4j.nn.conf.BackpropType
   org.deeplearning4j.nn.conf.Updater
   org.deeplearning4j.nn.conf.GradientNormalization
   org.deeplearning4j.nn.conf.layers.DenseLayer$Builder
   org.deeplearning4j.nn.conf.layers.ConvolutionLayer$Builder
   org.deeplearning4j.nn.conf.layers.LSTM$Builder
   org.deeplearning4j.nn.conf.layers.RnnOutputLayer$Builder
   org.deeplearning4j.nn.conf.layers.OutputLayer$Builder
   org.deeplearning4j.nn.conf.layers.SubsamplingLayer$Builder
   org.deeplearning4j.nn.conf.layers.SubsamplingLayer$PoolingType
   org.deeplearning4j.nn.weights.WeightInit
   org.deeplearning4j.optimize.listeners.ScoreIterationListener
   org.nd4j.linalg.learning.config.Adam
   org.nd4j.linalg.activations.Activation
   org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction
   org.nd4j.linalg.factory.Nd4j
   org.nd4j.linalg.indexing.NDArrayIndex
   org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor
   org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor
   org.deeplearning4j.eval.RegressionEvaluation
   org.deeplearning4j.eval.ROC
   org.nd4j.linalg.indexing.INDArrayIndex
   org.deeplearning4j.nn.graph.ComputationGraph
   org.datavec.api.split.FileSplit
   org.deeplearning4j.nn.conf.graph.MergeVertex
   
   java.io.File
   org.apache.commons.io.FilenameUtils
   org.apache.commons.io.FileUtils
   org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
   org.datavec.api.records.reader.impl.csv.CSVRecordReader
   org.datavec.api.records.reader.SequenceRecordReader
   org.datavec.api.split.NumberedFileInputSplit)
  (:gen-class))

(def opts
  {:clouds.dir/shell "/opt/app/"
   :clouds.dir/target "/opt/app/tmp/data/clouds/"})

(defn bash-script-fetch-data
  [{:clouds.dir/keys [target]}]
  (format "
  DIR=%s
  mkdir -p $DIR
  cd $DIR

  wget https://dl4jdata.blob.core.windows.net/training/tutorials/Cloud.tar.gz
  tar -xvzf Cloud.tar.gz
  mv ./Cloud/* ./
  " target))

(defn fetch-data
  [{:clouds.dir/keys [shell] :as opts}]
  (sh "bash" "-c" (bash-script-fetch-data opts)  :dir shell))

#_(:exit (fetch-data opts))

(def path (opts :clouds.dir/target))

(def train-base-dir1 (FilenameUtils/concat path "train/n1/train.csv"))
(def train-base-dir2 (FilenameUtils/concat path "train/n2/train.csv"))
(def train-base-dir3 (FilenameUtils/concat path "train/n3/train.csv"))
(def train-base-dir4 (FilenameUtils/concat path "train/n4/train.csv"))
(def train-base-dir5 (FilenameUtils/concat path "train/n5/train.csv"))

(def test-base-dir1 (FilenameUtils/concat path "test/n1/test.csv"))
(def test-base-dir2 (FilenameUtils/concat path "test/n2/test.csv"))
(def test-base-dir3 (FilenameUtils/concat path "test/n3/test.csv"))
(def test-base-dir4 (FilenameUtils/concat path "test/n4/test.csv"))
(def test-base-dir5 (FilenameUtils/concat path "test/n5/test.csv"))

(comment

  (do

    (def rr-train1 (CSVRecordReader. 1))
    (.initialize rr-train1 (FileSplit. (File. train-base-dir1)))
    (def rr-train2 (CSVRecordReader. 1))
    (.initialize rr-train2 (FileSplit. (File. train-base-dir2)))
    (def rr-train3 (CSVRecordReader. 1))
    (.initialize rr-train3 (FileSplit. (File. train-base-dir3)))
    (def rr-train4 (CSVRecordReader. 1))
    (.initialize rr-train4 (FileSplit. (File. train-base-dir4)))
    (def rr-train5 (CSVRecordReader. 1))
    (.initialize rr-train5 (FileSplit. (File. train-base-dir5)))

    (def train-iter (-> (RecordReaderMultiDataSetIterator$Builder. 20)
                        (.addReader "rr1" rr-train1)
                        (.addReader "rr2" rr-train2)
                        (.addReader "rr3" rr-train3)
                        (.addReader "rr4" rr-train4)
                        (.addReader "rr5" rr-train5)
                        (.addInput "rr1" 1 3)
                        (.addInput "rr2" 0 2)
                        (.addInput "rr3" 0 2)
                        (.addInput "rr4" 0 2)
                        (.addInput "rr5" 0 2)
                        (.addOutputOneHot "rr1" 0 2)
                        (.build)))

    (def rr-test1 (CSVRecordReader. 1))
    (.initialize rr-test1 (FileSplit. (File. test-base-dir1)))
    (def rr-test2 (CSVRecordReader. 1))
    (.initialize rr-test2 (FileSplit. (File. test-base-dir2)))
    (def rr-test3 (CSVRecordReader. 1))
    (.initialize rr-test3 (FileSplit. (File. test-base-dir3)))
    (def rr-test4 (CSVRecordReader. 1))
    (.initialize rr-test4 (FileSplit. (File. test-base-dir4)))
    (def rr-test5 (CSVRecordReader. 1))
    (.initialize rr-test5 (FileSplit. (File. test-base-dir5)))

    (def test-iter (-> (RecordReaderMultiDataSetIterator$Builder. 20)
                       (.addReader "rr1" rr-test1)
                       (.addReader "rr2" rr-test2)
                       (.addReader "rr3" rr-test3)
                       (.addReader "rr4" rr-test4)
                       (.addReader "rr5" rr-test5)
                       (.addInput "rr1" 1 3)
                       (.addInput "rr2" 0 2)
                       (.addInput "rr3" 0 2)
                       (.addInput "rr4" 0 2)
                       (.addInput "rr5" 0 2)
                       (.addOutputOneHot "rr1" 0 2)
                       (.build)))

    ;
    )

  (->> MergeVertex
       (.getDeclaredConstructors)
       (map #(.toGenericString %))
       (mapv #(string/split % #",")))

  (def conf (-> (NeuralNetConfiguration$Builder.)
                (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                (.updater Updater/ADAM)
                (.graphBuilder)
                (.addInputs (into-array ["input1" "input2" "input3" "input4" "input5"]))
                (.addLayer "L1" (-> (DenseLayer$Builder.)
                                    (.weightInit WeightInit/XAVIER)
                                    (.activation Activation/RELU)
                                    (.nIn 3)
                                    (.nOut 50)
                                    (.build)) (into-array ["input1"]))
                (.addLayer "L2" (-> (DenseLayer$Builder.)
                                    (.weightInit WeightInit/XAVIER)
                                    (.activation Activation/RELU)
                                    (.nIn 3)
                                    (.nOut 50)
                                    (.build)) (into-array ["input2"]))
                (.addLayer "L3" (-> (DenseLayer$Builder.)
                                    (.weightInit WeightInit/XAVIER)
                                    (.activation Activation/RELU)
                                    (.nIn 3)
                                    (.nOut 50)
                                    (.build)) (into-array ["input3"]))
                (.addLayer "L4" (-> (DenseLayer$Builder.)
                                    (.weightInit WeightInit/XAVIER)
                                    (.activation Activation/RELU)
                                    (.nIn 3)
                                    (.nOut 50)
                                    (.build)) (into-array ["input4"]))
                (.addLayer "L5" (-> (DenseLayer$Builder.)
                                    (.weightInit WeightInit/XAVIER)
                                    (.activation Activation/RELU)
                                    (.nIn 3)
                                    (.nOut 50)
                                    (.build)) (into-array ["input5"]))
                (.addVertex "merge"
                            (MergeVertex.)
                            (into-array ["L1" "L2" "L3" "L4" "L5"]))
                (.addLayer "L6" (-> (DenseLayer$Builder.)
                                    (.weightInit WeightInit/XAVIER)
                                    (.activation Activation/RELU)
                                    (.nIn 250)
                                    (.nOut 125)
                                    (.build)) (into-array ["merge"]))
                (.addLayer "out" (-> (OutputLayer$Builder.)
                                     (.lossFunction LossFunctions$LossFunction/MCXENT)
                                     (.weightInit WeightInit/XAVIER)
                                     (.activation Activation/SOFTMAX)
                                     (.nIn 125)
                                     (.nOut 2)
                                     (.build)) (into-array ["L6"]))
                (.setOutputs (into-array ["out"]))
                (.build)))

  (def model (ComputationGraph. conf))
  (.init model)

  (def fu (future-call (fn []
                         (prn "--started training")
                         (doseq [epoch (range 0 5)
                                 :while (not (.isInterrupted (Thread/currentThread)))]
                           (time
                            (do
                              (.fit model train-iter)
                              (prn (str "epoch " epoch " complete"))))) ; ~250s
                         (prn "--finished training"))))
  
  (future-cancel fu)

  (def roc (time (.evaluateROC model test-iter 100)))

  (println "FINAL TEST AUC: " (.calculateAUC roc))

  ;
  )


