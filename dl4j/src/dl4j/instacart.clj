(ns dl4j.instacart
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

   org.apache.commons.io.FilenameUtils
   org.apache.commons.io.FileUtils
   org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
   org.datavec.api.records.reader.SequenceRecordReader
   org.datavec.api.split.NumberedFileInputSplit)
  (:gen-class))

(def opts
  {:instacart.dir/shell "/opt/app/"
   :instacart.dir/target "/opt/app/tmp/data/instacart/"})

(defn bash-script-fetch-data
  [{:instacart.dir/keys [target]}]
  (format "
  DIR=%s
  mkdir -p $DIR
  cd $DIR

  wget https://dl4jdata.blob.core.windows.net/training/tutorials/instacart.tar.gz
  tar -xvzf instacart.tar.gz
  mv ./instacart/* ./
  " target))

(defn fetch-data
  [{:instacart.dir/keys [shell] :as opts}]
  (sh "bash" "-c" (bash-script-fetch-data opts)  :dir shell))

#_(:exit (fetch-data opts))

(def path (opts :instacart.dir/target))
(def features-base-dir (FilenameUtils/concat path "features"))
(def targets-base-dir (FilenameUtils/concat path "breakfast"))
(def auxil-base-dir (FilenameUtils/concat path "dairy"))

(comment
  
  ;; multitask

  (do

    (def train-features (CSVSequenceRecordReader. 1 ","))
    (.initialize train-features
                 (NumberedFileInputSplit. (str features-base-dir "/%d.csv") 1 4000))

    (def train-breakfast (CSVSequenceRecordReader. 1 ","))
    (.initialize train-breakfast
                 (NumberedFileInputSplit. (str targets-base-dir "/%d.csv") 1 4000))

    (def train-dairy (CSVSequenceRecordReader. 1 ","))
    (.initialize train-dairy
                 (NumberedFileInputSplit. (str auxil-base-dir "/%d.csv") 1 4000))

    (def train-iter (-> (RecordReaderMultiDataSetIterator$Builder. 20)
                        (.addSequenceReader "rr1" train-features)
                        (.addInput "rr1")
                        (.addSequenceReader "rr2" train-breakfast)
                        (.addOutput "rr2")
                        (.addSequenceReader "rr3" train-dairy)
                        (.addOutput "rr3")
                        (.sequenceAlignmentMode RecordReaderMultiDataSetIterator$AlignmentMode/ALIGN_END)
                        (.build)))

    (def test-features (CSVSequenceRecordReader. 1 ","))
    (.initialize train-features
                 (NumberedFileInputSplit. (str features-base-dir "/%d.csv") 4001 5000))

    (def test-breakfast (CSVSequenceRecordReader. 1 ","))
    (.initialize train-breakfast
                 (NumberedFileInputSplit. (str targets-base-dir "/%d.csv") 4001 5000))

    (def test-dairy (CSVSequenceRecordReader. 1 ","))
    (.initialize train-dairy
                 (NumberedFileInputSplit. (str auxil-base-dir "/%d.csv") 4001 5000))


    (def test-iter (-> (RecordReaderMultiDataSetIterator$Builder. 20)
                       (.addSequenceReader "rr1" train-features)
                       (.addInput "rr1")
                       (.addSequenceReader "rr2" train-breakfast)
                       (.addOutput "rr2")
                       (.addSequenceReader "rr3" train-dairy)
                       (.addOutput "rr3")
                       (.sequenceAlignmentMode RecordReaderMultiDataSetIterator$AlignmentMode/ALIGN_END)
                       (.build)))

    ;
    )

  (def conf (-> (NeuralNetConfiguration$Builder.)
                (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                (.seed 12345)
                (.weightInit WeightInit/XAVIER)
                (.dropOut 0.25)
                (.graphBuilder)
                (.addInputs (into-array ["input"]))
                (.addLayer "L1" (-> (LSTM$Builder.)
                                    (.nIn 134)
                                    (.nOut 150)
                                    (.updater Updater/ADAM)
                                    (.gradientNormalization GradientNormalization/ClipElementWiseAbsoluteValue)
                                    (.gradientNormalizationThreshold 10)
                                    (.activation Activation/TANH)
                                    (.build)) (into-array ["input"]))
                (.addLayer "out1" (-> (RnnOutputLayer$Builder. LossFunctions$LossFunction/XENT)
                                      (.updater Updater/ADAM)
                                      (.gradientNormalization GradientNormalization/ClipElementWiseAbsoluteValue)
                                      (.gradientNormalizationThreshold 10)
                                      (.activation Activation/SIGMOID)
                                      (.nIn 150)
                                      (.nOut 1)
                                      (.build)) (into-array ["L1"]))
                (.addLayer "out2" (-> (RnnOutputLayer$Builder. LossFunctions$LossFunction/XENT)
                                      (.updater Updater/ADAM)
                                      (.gradientNormalization GradientNormalization/ClipElementWiseAbsoluteValue)
                                      (.gradientNormalizationThreshold 10)
                                      (.activation Activation/SIGMOID)
                                      (.nIn 150)
                                      (.nOut 1)
                                      (.build)) (into-array ["L1"]))
                (.setOutputs (into-array ["out1" "out2"]))
                (.build)))

  (json/read-str (.toJson conf) :key-fn keyword)
  (json/read-str (.toJson conf) :key-fn keyword)

  (def net (ComputationGraph. conf))
  (do (.init net))

  (doseq [epoch (range 0 5)]
    (time
     (do
       (.fit net train-iter)
       (.reset train-iter)
       (prn (str "epoch " epoch " complete"))))) ; ~28s


  (def roc (ROC.))

  (.reset test-iter)

  (def a (Nd4j/ones  2 2))
  (.getDouble a (int-array [1]))

  (def pik (atom nil))
  (aget @pik 0)
  
  (while (.hasNext test-iter)
    (let [next (.next test-iter)
          features (.getFeatures next)
          _ (reset! pik features)
          output (.output  net (into-array [(aget features 0)]) )
          labels (.getLabels next)]
      (.evalTimeSeries roc (aget labels 0) (aget output 0))))
  
  (println (.calculateAUC roc)) ; 0.86





  ;
  )


(comment

  ;; single task

  (do
    (def train-features (CSVSequenceRecordReader. 1 ","))
    (.initialize train-features
                 (NumberedFileInputSplit. (str features-base-dir "/%d.csv") 1 4000))
    (def train-labels (CSVSequenceRecordReader. 1 ","))
    (.initialize train-labels
                 (NumberedFileInputSplit. (str features-base-dir "/%d.csv") 1 4000))

    (def test-features (CSVSequenceRecordReader. 1 ","))
    (.initialize test-features
                 (NumberedFileInputSplit. (str features-base-dir "/%d.csv") 4001 5000))
    (def test-labels (CSVSequenceRecordReader. 1 ","))
    (.initialize test-labels
                 (NumberedFileInputSplit. (str features-base-dir "/%d.csv") 4001 5000))

    (def train-iter (SequenceRecordReaderDataSetIterator.
                     train-features
                     train-labels
                     32
                     2
                     false
                     SequenceRecordReaderDataSetIterator$AlignmentMode/ALIGN_END))

    (def test-iter (SequenceRecordReaderDataSetIterator.
                    test-features
                    test-labels
                    32
                    2
                    false
                    SequenceRecordReaderDataSetIterator$AlignmentMode/ALIGN_END)))

  (def conf (-> (NeuralNetConfiguration$Builder.)
                (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                (.seed 12345)
                (.dropOut 0.25)
                (.weightInit WeightInit/XAVIER)
                (.updater Updater/ADAM)
                (.list)
                (.layer 0 (-> (LSTM$Builder.)
                              (.activation Activation/TANH)
                              (.gradientNormalization GradientNormalization/ClipElementWiseAbsoluteValue)
                              (.gradientNormalizationThreshold 10)
                              (.nIn 134)
                              (.nOut 150)
                              (.build)))
                (.layer 1 (-> (RnnOutputLayer$Builder. LossFunctions$LossFunction/XENT)
                              (.activation Activation/SOFTMAX)
                              (.nIn 150)
                              (.nOut 2)
                              (.build)))
                (.validateOutputLayerConfig false) ; not recommended
                (.build)))

  (def net (MultiLayerNetwork. conf))
  (do (.init net))

  (doseq [epoch (range 0 5)]
    (time
     (do
       (.fit net train-iter)
       (.reset train-iter)
       (prn (str "epoch " epoch " complete")))))

  (def roc (ROC. 100))

  (do
    (.reset test-iter)
    (while (.hasNext test-iter)
      (let [next (.next test-iter)
            features (.getFeatures next)
            output (.output net features)]
        (.evalTimeSeries roc (.getLabels next) output))))

  (println (.calculateAUC roc)) 
  ; 0.97
  ; seems too high..


  ;
  )