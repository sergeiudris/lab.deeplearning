(ns dl4j.paragraph-vec
  (:require [clojure.string :as string]
            [pad.prn.core :refer [bash-ls]])
  (:import
   org.deeplearning4j.examples.download.DownloaderUtility))


(comment

  (.Download DownloaderUtility/NLPDATA)
  ; "/root/dl4j-examples-data/dl4j-examples/nlp"

  (bash-ls "/root")
  

  ;
  )