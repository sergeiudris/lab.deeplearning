(ns dl4j.doc2vec-cls
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
   org.deeplearning4j.models.word2vec.VocabWord
   org.deeplearning4j.text.documentiterator.FileLabelAwareIterator
   org.deeplearning4j.text.documentiterator.FileLabelAwareIterator$Builder
   org.deeplearning4j.text.documentiterator.LabelAwareIterator
   org.deeplearning4j.text.documentiterator.LabelledDocument
   org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
   org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
   org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
   org.nd4j.linalg.api.ndarray.INDArray
   org.nd4j.linalg.primitives.Pair
   org.slf4j.Logger
   org.slf4j.LoggerFactory

   java.io.File
   java.io.IOException
   java.util.List
   ))

 #_(.Download DownloaderUtility/NLPDATA)
  ; "/root/dl4j-examples-data/dl4j-examples/nlp"

(comment

  (def data-local-path "/root/dl4j-examples-data/dl4j-examples/nlp")

  (def resource (File. data-local-path "paravec/labeled"))
  (def iterator (-> (FileLabelAwareIterator$Builder.)
                    (.addSourceFolder resource)
                    (.build)))
  (def tokenizer-factory (DefaultTokenizerFactory.))
  (.setTokenPreProcessor tokenizer-factory (CommonPreprocessor.))

  (def paragraph-vectors (-> (ParagraphVectors$Builder.)
                             (.learningRate 0.025)
                             (.minLearningRate 0.001)
                             (.batchSize 1000)
                             (.epochs 20)
                             (.iterate iterator)
                             (.trainWordVectors true)
                             (.tokenizerFactory tokenizer-factory)
                             (.build)))

  
  (.fit paragraph-vectors)

  (def path-to-save "/opt/app/tmp/doc2vec.zip")
  (WordVectorSerializer/writeParagraphVectors paragraph-vectors path-to-save)
  (def paragraph-vectors (WordVectorSerializer/readParagraphVectors path-to-save))

  (def unclassified-resource (File. data-local-path "paravec/unlabeled"))
  (def unclassified-iterator (-> (FileLabelAwareIterator$Builder.)
                                 (.addSourceFolder unclassified-resource)
                                 (.build)))

  (def means-builder (MeansBuilder.
                      (.getLookupTable paragraph-vectors) ; (InMemoryLookupTable<VocabWord>)
                      tokenizer-factory))

  (def seeker (LabelSeeker.
               (.. iterator getLabelsSource getLabels)
               (.getLookupTable paragraph-vectors) ; (InMemoryLookupTable<VocabWord>)
               ))

  (while (.hasNextDocument unclassified-iterator)
    (let [document (.nextDocument unclassified-iterator)
          document-as-centroid (.documentAsVector means-builder document)
          scores (.getScores seeker document-as-centroid)]
      (prn (str "document " (.getLabels document) " falls into categories:"))
      (doseq [score (iterator-seq (.iterator scores))]
        (prn (str (.getFirst score) ":" (.getSecond score))))))

  ;
  )