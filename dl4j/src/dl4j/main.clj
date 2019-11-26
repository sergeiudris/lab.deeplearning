(ns dl4j.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [dl4j.core]
            [dl4j.emnist]
            [dl4j.tutorial]
            [dl4j.mnist-anomaly]
            [dl4j.facenet]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server :port 7788))