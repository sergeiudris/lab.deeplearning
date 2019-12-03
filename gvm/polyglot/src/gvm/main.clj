(ns gvm.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [gvm.polyglot]
   ;
            )
  (:gen-class))

(defn -main  [& args]
  (start-nrepl-server "0.0.0.0" 7788))