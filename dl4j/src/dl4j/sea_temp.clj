(ns dl4j.sea-temp
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
   org.nd4j.linalg.indexing.INDArrayIndex

   org.apache.commons.io.FilenameUtils
   org.apache.commons.io.FileUtils
   org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
   org.datavec.api.records.reader.SequenceRecordReader
   org.datavec.api.split.NumberedFileInputSplit)
  (:gen-class))

(def opts
  {:sea.dir/shell "/opt/app/"
   :sea.dir/target "/opt/app/tmp/data/sea/"})

(defn bash-script-fetch-data
  [{:sea.dir/keys [target]}]
  (format "
  DIR=%s
  mkdir -p $DIR
  cd $DIR

  wget https://dl4jdata.blob.core.windows.net/training/seatemp/sea_temp.tar.gz
  tar -xvzf sea_temp.tar.gz
  mv ./sea_temp/* ./
  " target))

(defn fetch-data
  [{:sea.dir/keys [shell] :as opts}]
  (sh "bash" "-c" (bash-script-fetch-data opts)  :dir shell))

#_(:exit (fetch-data opts))

(def path (opts :sea.dir/target))
(def features-base-dir (FilenameUtils/concat path "features"))
(def targets-base-dir (FilenameUtils/concat path "targets"))
(def futures-base-dir (FilenameUtils/concat path "futures"))

(def num-skip-lines 1)
(def regression true)
(def batch-size 32)


(comment


  (do

    (def train-features (CSVSequenceRecordReader. num-skip-lines ","))
    (.initialize train-features
                 (NumberedFileInputSplit. (str features-base-dir "/%d.csv") 1 1600 #_1936))
    (def train-targets (CSVSequenceRecordReader. num-skip-lines ","))
    (.initialize train-targets
                 (NumberedFileInputSplit. (str targets-base-dir "/%d.csv") 1 1600 #_1936))

    (def train-iter (SequenceRecordReaderDataSetIterator.
                     train-features
                     train-targets
                     batch-size
                     10
                     regression
                     SequenceRecordReaderDataSetIterator$AlignmentMode/EQUAL_LENGTH))

    (def test-features (CSVSequenceRecordReader. num-skip-lines ","))
    (.initialize test-features
                 (NumberedFileInputSplit. (str features-base-dir "/%d.csv") 1601 1736 #_1937 #_2089))
    (def test-targets (CSVSequenceRecordReader. num-skip-lines ","))
    (.initialize test-targets
                 (NumberedFileInputSplit. (str targets-base-dir "/%d.csv") 1601 1736 #_1937 #_2089))

    (def test-iter (SequenceRecordReaderDataSetIterator.
                    test-features
                    test-targets
                    batch-size
                    10
                    regression
                    SequenceRecordReaderDataSetIterator$AlignmentMode/EQUAL_LENGTH))

    (def future-features (CSVSequenceRecordReader. num-skip-lines ","))
    (.initialize future-features
                 (NumberedFileInputSplit. (str futures-base-dir "/%d.csv") 1601 1736 #_1936))
    (def future-labels (CSVSequenceRecordReader. num-skip-lines ","))
    (.initialize future-labels
                 (NumberedFileInputSplit. (str futures-base-dir "/%d.csv") 1601 1736 #_1936))

    (def future-iter (SequenceRecordReaderDataSetIterator.
                      future-features
                      future-labels
                      batch-size
                      10
                      regression
                      SequenceRecordReaderDataSetIterator$AlignmentMode/EQUAL_LENGTH)))

  ; part 1

  (do
    (def v-height 13)
    (def v-width 4)
    (def kernel-size 2)
    (def num-channels 1))

  (def conf (-> (NeuralNetConfiguration$Builder.)
                (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                (.seed 12345)
                (.weightInit WeightInit/XAVIER)
                (.list)
                (.layer 0 (-> (ConvolutionLayer$Builder.
                               (int-array [kernel-size kernel-size]))
                              (.updater Updater/ADAGRAD)
                              (.nIn num-channels)
                              (.nOut 7)
                              (.stride (int-array [2 2]))
                              (.activation Activation/RELU)
                              (.build)))
                (.layer 1 (-> (LSTM$Builder.)
                              (.activation Activation/SOFTSIGN)
                              (.nIn 84)
                              (.nOut 200)
                              (.updater Updater/ADAGRAD)
                              (.gradientNormalization GradientNormalization/ClipElementWiseAbsoluteValue)
                              (.gradientNormalizationThreshold 10)
                              (.build)))
                (.layer 2 (-> (RnnOutputLayer$Builder. LossFunctions$LossFunction/MSE)
                              (.activation Activation/IDENTITY)
                              (.nIn 200)
                              (.updater Updater/ADAGRAD)
                              (.nOut 52)
                              (.gradientNormalization GradientNormalization/ClipElementWiseAbsoluteValue)
                              (.gradientNormalizationThreshold 10)
                              (.build)))
                (.inputPreProcessor (int 0) (RnnToCnnPreProcessor. v-height v-width num-channels))
                (.inputPreProcessor (int 1) (CnnToRnnPreProcessor. 6 2 7))
                (.build)))
  (json/read-str (.toJson conf) :key-fn keyword)

  (def net (MultiLayerNetwork. conf))
  (.init net)

  (.reset train-iter)
  (doseq [epoch (range 0 25)]
    (time
     (do
       (.fit net train-iter)
       (prn "epoch " epoch " complete")))) ; ~19s

  (.score net)

  (def evaluation (.evaluateRegression net test-iter))
  (.reset test-iter)
  (println (.stats evaluation))

  ; part 2

  (do
    (def v-height 13)
    (def v-width 4)
    (def kernel-size 2)
    (def num-channels 1))

  (def conf (-> (NeuralNetConfiguration$Builder.)
                (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                (.seed 12345)
                (.updater Updater/ADAM)
                (.weightInit WeightInit/XAVIER)
                (.list)
                (.layer 0 (-> (ConvolutionLayer$Builder.
                               (int-array [kernel-size kernel-size]))
                              (.updater Updater/ADAGRAD)
                              (.nIn num-channels)
                              (.nOut 7)
                              (.stride (int-array [2 2]))
                              (.activation Activation/RELU)
                              (.build)))
                (.layer 1 (-> (SubsamplingLayer$Builder. SubsamplingLayer$PoolingType/MAX)
                              (.kernelSize (int-array [kernel-size kernel-size]))
                              (.stride (int-array [2 2]))
                              (.build)))
                (.layer 2 (-> (LSTM$Builder.)
                              (.activation Activation/SOFTSIGN)
                              (.nIn 21)
                              (.nOut 100)
                              (.updater Updater/ADAGRAD)
                              (.gradientNormalization GradientNormalization/ClipElementWiseAbsoluteValue)
                              (.gradientNormalizationThreshold 10)
                              (.build)))
                (.layer 3 (-> (RnnOutputLayer$Builder. LossFunctions$LossFunction/MSE)
                              (.activation Activation/IDENTITY)
                              (.nIn 100)
                              (.updater Updater/ADAGRAD)
                              (.nOut 52)
                              (.gradientNormalization GradientNormalization/ClipElementWiseAbsoluteValue)
                              (.gradientNormalizationThreshold 10)
                              (.build)))
                (.inputPreProcessor (int 0) (RnnToCnnPreProcessor. v-height v-width num-channels))
                (.inputPreProcessor (int 2) (CnnToRnnPreProcessor. 3 1 7))
                (.build)))

  (json/read-str (.toJson conf) :key-fn keyword)

  (def net (MultiLayerNetwork. conf))
  (.init net)

  (.reset train-iter)
  (doseq [epoch (range 0 15)]
    (time
     (do
       (.fit net train-iter)
       (prn "epoch " epoch " complete"))))

  (def evaluation (RegressionEvaluation.))



  (do
    (.reset test-iter)
    (.reset future-iter)
    (while (.hasNext test-iter)
      (let [next (.next test-iter)
            features (.getFeatures next)
            pred (atom (Nd4j/zeros 1 2))]
        (doseq [i (range 0 50)]
          (reset! pred (.rnnTimeStep net (.get features
                                               (into-array INDArrayIndex
                                                           [(NDArrayIndex/all)
                                                            (NDArrayIndex/all)
                                                            (NDArrayIndex/interval i (inc i))])))))
        (let [correct (.next future-iter)
              c-features (.getFeatures correct)]
          (doseq [i (range 0 10)]
            (.evalTimeSeries evaluation
                             @pred
                             (.get c-features
                                   (into-array INDArrayIndex
                                               [(NDArrayIndex/all)
                                                (NDArrayIndex/all)
                                                (NDArrayIndex/interval i (inc i))])))
            (reset! pred (.rnnTimeStep net @pred)))))
      (.rnnClearPreviousState net)))
  
  (println (.stats evaluation))
  
  



  ;
  )