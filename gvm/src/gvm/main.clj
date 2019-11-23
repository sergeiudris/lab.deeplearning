(ns gvm.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [gvm.core]
            [gvm.polyglot]
   ;
            )
  (:gen-class))

(defn -main  [& args]
  (start-nrepl-server)
  (gvm.core/ping))