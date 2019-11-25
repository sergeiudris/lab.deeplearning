(ns dl4j.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [dl4j.core]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server :port 7788))