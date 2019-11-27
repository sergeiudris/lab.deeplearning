(ns dl4j.word2vec
  (:require [clojure.string :as string]
            [pad.prn.core :refer [bash-ls]])
  (:import
   org.deeplearning4j.examples.download.DownloaderUtility
   org.deeplearning4j.models.word2vec.Word2Vec$Builder
   org.deeplearning4j.text.sentenceiterator.BasicLineIterator
   org.deeplearning4j.text.sentenceiterator.SentenceIterator
   org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
   org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
   org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
   org.slf4j.Logger
   org.slf4j.LoggerFactory

   java.io.File
   java.util.Collection))




(comment

  (def data-local-path (.Download DownloaderUtility/NLPDATA))
  (bash-ls "/root")
  
  (def file-path (.getAbsolutePath (File. data-local-path "raw_sentences.txt")))
  
  (def iter (BasicLineIterator. file-path))
  
  (def t (DefaultTokenizerFactory.))
  
  ; CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
  ; So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
  ; Additionally it forces lower case for all tokens.
  
  (.setTokenPreProcessor t (CommonPreprocessor.))
  
  (def vec (-> (Word2Vec$Builder.)
               (.minWordFrequency 5)
               (.iterations 1)
               (.layerSize 100)
               (.seed 42)
               (.windowSize 5)
               (.iterate iter)
               (.tokenizerFactory t)
               (.build)
               ))
  
  (.fit vec)
  
  (def lst (.wordsNearestSum vec "day" 10))
  ; Cannot perform in-place operation "muli": 
  ; result array shape does not match the broadcast operation output shape: [100].muli([1, 100]) != [100].
  ; fix
  ; https://github.com/eclipse/deeplearning4j/issues/8288#issuecomment-554167328
  
  ; bottomline: need to wait for the next version or use snapshots
  


  ;
  )