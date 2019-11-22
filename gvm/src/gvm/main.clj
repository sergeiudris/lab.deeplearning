(ns gvm.main
  (:require #_[pad.nrepl.core :refer [start-nrepl-server]]
            [gvm.core]
   ;
            )
  (:gen-class))

(defn -main  [& args]
  #_(start-nrepl-server)
  (gvm.core/ping))