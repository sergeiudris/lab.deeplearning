(ns dl4j.doc2vec-infer
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
   org.nd4j.linalg.ops.transforms.Transforms

   java.io.File;
   java.io.IOException;
   java.util.List;
   ))

 #_(.Download DownloaderUtility/NLPDATA)
  ; "/root/dl4j-examples-data/dl4j-examples/nlp"

(comment 
  
  (def data-local-path "/root/dl4j-examples-data/dl4j-examples/nlp")
  
  (def resource (File. data-local-path "paravec/simple.pv"))
  
  (def tokenizer-factory (DefaultTokenizerFactory.))
  
  (.setTokenPreProcessor tokenizer-factory (CommonPreprocessor.))
  
  (def vectors (WordVectorSerializer/readParagraphVectors resource))
  (.setTokenizerFactory vectors tokenizer-factory)
  
  (.. vectors (getConfiguration) (setIteration 1))
  
  (def inferred-vector-a (.inferVector vectors "This is my world .") )
  (def inferred-vector-a2 (.inferVector vectors "This is my world ."))
  (def inferred-vector-b (.inferVector vectors "This is my way ."))

  ; high similarity expected here,
  ; since in underlying corpus words WAY and WORLD have really close context 
  (Transforms/cosineSim  inferred-vector-a inferred-vector-b)
  
  ; equality expected here, since inference is happening for the same sentences
  (Transforms/cosineSim inferred-vector-a inferred-vector-a2)
  
  (Transforms/cosineSim
   (.inferVector vectors "That's what I expected to find .")
   (.inferVector vectors "This is what I was looking for ."))
  
  (def x (.inferVector vectors "That's what I expected to find ."))
  (.shapeDescriptor x)
  
  (def s (string/join \space (repeat 500 "That's what I expected to find ")))
  (count s)
  (def x (.inferVector vectors s ))
  (.shapeDescriptor x)
  ;
  )