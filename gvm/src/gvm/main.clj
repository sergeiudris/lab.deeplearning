(ns gvm.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [gvm.core]
            [d2l.nlp]
            [d2l.arxiv]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server))