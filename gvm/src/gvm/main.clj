(ns gvm.main
  (:require #_[pad.nrepl.core :refer [start-nrepl-server]]
            [gvm.core]
            [d2l.nlp]
            [d2l.arxiv]
   ;
            )
  (:gen-class))

(defn -main  [& args]
  #_(start-nrepl-server)
  (gvm.core/ping))