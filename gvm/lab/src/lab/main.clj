(ns lab.main
  (:require #_[pad.nrepl.core :refer [start-nrepl-server]]
            [lab.core]
            [d2l.nlp]
            [d2l.arxiv]
   ;
            )
  (:gen-class))

(defn -main  [& args]
  #_(start-nrepl-server)
  (lab.core/ping))