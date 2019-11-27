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

   java.io.File;
   java.io.IOException;
   java.util.List;
   ))

 #_(.Download DownloaderUtility/NLPDATA)
  ; "/root/dl4j-examples-data/dl4j-examples/nlp"

(comment

  (def data-local-path "/root/dl4j-examples-data/dl4j-examples/nlp")

  (def file (File. data-local-path "raw_sentences.txt"))

  (def iter (BasicLineIterator. file))

  (def tokenizer-factory (DefaultTokenizerFactory.))

  (.setTokenPreProcessor tokenizer-factory (CommonPreprocessor.))

  (def cache (AbstractCache.))

  (def source (LabelsSource. "DOC_"))

  (def par-vec (-> (ParagraphVectors$Builder.)
                   (.minWordFrequency 1)
                   (.iterations 5)
                   (.epochs 1)
                   (.layerSize 100)
                   (.learningRate 0.025)
                   (.labelsSource source)
                   (.windowSize 5)
                   (.iterate iter)
                   (.trainWordVectors false)
                   (.vocabCache cache)
                   (.tokenizerFactory tokenizer-factory)
                   (.sampling 0.0)
                   (.build)))

  (time
   (.fit par-vec)) ; 875169.137487 msecs ~15min

  ; In training corpus we have few lines that contain pretty close words invloved.
  ;           These sentences should be pretty close to each other in vector space
  ;           line 3721: This is my way .
  ;           line 6348: This is my case .
  ;           line 9836: This is my house .
  ;           line 12493: This is my world .
  ;           line 16393: This is my work .
  ;           this is special sentence, that has nothing common with previous sentences
  ;           line 9853: We now have one .
  ; Note that docs are indexed from 0

  (def path-to-save "/opt/app/tmp/doc2vec_text.zip")
  (WordVectorSerializer/writeParagraphVectors par-vec path-to-save)
  (def par-vec2 (WordVectorSerializer/readParagraphVectors path-to-save))

  ; 'This is my house .'/'This is my world .'
  (.similarity par-vec "DOC_9835" "DOC_12492")
  ; 0.7735093235969543

  ; 'This is my way .'/'This is my work .'
  (.similarity par-vec "DOC_3720" "DOC_16392")
  ; 0.4487181603908539

  ; 'This is my case .'/'This is my way .'
  (.similarity par-vec "DOC_6347" "DOC_3720")
  ; 0.8837857842445374

  ; likelihood in this case should be significantly lower
  (.similarity par-vec "DOC_3721" "DOC_9853")
  ; 0.12685196101665497

  ;
  )