(ns dl4j.main
  (:require [dl4j.nrepl :refer [start-nrepl-server]]
            [dl4j.core]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server :port 7788))