(ns dl4j.word2vec
  (:require [clojure.string :as string]
            [pad.prn.core :refer [bash-ls]])
  (:import
   org.deeplearning4j.examples.download.DownloaderUtility
   org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
   org.deeplearning4j.models.word2vec.Word2Vec$Builder
   org.deeplearning4j.text.sentenceiterator.BasicLineIterator
   org.deeplearning4j.text.sentenceiterator.SentenceIterator
   org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
   org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
   org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
   org.slf4j.Logger
   org.slf4j.LoggerFactory
   org.nd4j.linalg.factory.Nd4j

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

  (def word2vec (-> (Word2Vec$Builder.)
                    (.minWordFrequency 5)
                    (.iterations 1)
                    (.layerSize 100)
                    (.seed 42)
                    (.windowSize 5)
                    (.iterate iter)
                    (.tokenizerFactory t)
                    (.build)))

  (.fit word2vec)

   ; save/load

  (def path-to-save "/opt/app/tmp/word2vec.txt")
  (WordVectorSerializer/writeWord2VecModel word2vec path-to-save)
  (def word2vec (WordVectorSerializer/readWord2VecModel path-to-save))

  (def lst (.wordsNearestSum word2vec "day" 10))
  ; Cannot perform in-place operation "muli": 
  ; result array shape does not match the broadcast operation output shape: [100].muli([1, 100]) != [100].
  ; fix
  ; https://github.com/eclipse/deeplearning4j/issues/8288#issuecomment-554167328

  ; bottomline: need to wait for the next version or use snapshots


  (def cos-sim (.similarity word2vec "day" "night"))

  (def king-list (.wordsNearest word2vec
                                (java.util.Arrays/asList (object-array ["king" "woman"]))
                                (java.util.Arrays/asList (object-array ["queen"]))
                                10))


  (def weight-lookup-table (.lookupTable word2vec))
  (def vectors (.vectors weight-lookup-table))
  (def word-vector-matrix (.getWordVectorMatrix word2vec "hello"))
  (def word-vector (.getWordVector word2vec "more"))


  ;
  )