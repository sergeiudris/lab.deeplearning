(ns dl4j.glove
  (:require [clojure.string :as string]
            [pad.prn.core :refer [bash-ls]])
  (:import
   org.deeplearning4j.examples.download.DownloaderUtility
   org.deeplearning4j.examples.nlp.paragraphvectors.tools.LabelSeeker
   org.deeplearning4j.examples.nlp.paragraphvectors.tools.MeansBuilder
   org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
   org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
   org.deeplearning4j.models.paragraphvectors.ParagraphVectors
   org.deeplearning4j.models.paragraphvectors.ParagraphVectors$Builder
   org.deeplearning4j.models.glove.Glove$Builder
   org.deeplearning4j.models.word2vec.VocabWord
   org.deeplearning4j.text.sentenceiterator.BasicLineIterator
   org.deeplearning4j.text.sentenceiterator.SentenceIterator
   org.deeplearning4j.text.documentiterator.FileLabelAwareIterator
   org.deeplearning4j.text.documentiterator.FileLabelAwareIterator$Builder
   org.deeplearning4j.text.documentiterator.LabelAwareIterator
   org.deeplearning4j.text.documentiterator.LabelsSource
   org.deeplearning4j.text.documentiterator.LabelledDocument
   org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
   org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
   org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
   org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache
   org.nd4j.linalg.api.ndarray.INDArray
   org.nd4j.linalg.primitives.Pair
   org.slf4j.Logger
   org.slf4j.LoggerFactory

   java.io.File
   java.io.IOException
   java.util.List))

#_(.Download DownloaderUtility/NLPDATA)
  ; "/root/dl4j-examples-data/dl4j-examples/nlp"


(comment
  (do
    (def data-local-path "/root/dl4j-examples-data/dl4j-examples/nlp")

    (def file (File. data-local-path "raw_sentences.txt"))

    (def iter (BasicLineIterator. file))

    (def tokenizer-factory (DefaultTokenizerFactory.))

    (.setTokenPreProcessor tokenizer-factory (CommonPreprocessor.)))

  (def glove (-> (Glove$Builder.)
                 (.iterate iter)
                 (.tokenizerFactory tokenizer-factory)
                 (.alpha 0.75)
                 (.learningRate 0.1)
                 (.epochs 25)
                 (.xMax 100)
                 (.batchSize 1000)
                 (.shuffle true)
                 (.symmetric true)
                 (.build)))

  (def fu (future-call (fn []
                         (prn "--started training")
                         (.fit glove)
                         (prn "--finished training"))))

  (.similarity glove "day" "night")
  (.nearestWords glove "day" 10)

  ; Exception in thread "VectorCalculationsThread 0" java.lang.RuntimeException: java.lang.UnsupportedOperationException
  ; at org.deeplearning4j.models.sequencevectors.SequenceVectors$VectorCalculationsThread.run(SequenceVectors.java:1343)

  ;
  )