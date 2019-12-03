(ns d2l.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [d2l.core]
            [d2l.mnist]
            [d2l.house]
            [d2l.mxnet-made-simple]
            [d2l.viz]
            [d2l.inception]
            [d2l.arxiv]
            [d2l.recom]
            [d2l.nlp]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server "0.0.0.0" 7788))