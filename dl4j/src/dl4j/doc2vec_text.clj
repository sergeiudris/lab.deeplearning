(ns dl4j.doc2vec-text
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

   java.io.File;
   java.io.IOException;
   java.util.List;
   ))

(comment
  
  
  ;
  )