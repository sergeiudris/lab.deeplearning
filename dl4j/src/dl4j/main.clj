(ns dl4j.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [dl4j.linear]
            [dl4j.emnist]
            [dl4j.tutorial]
            [dl4j.mnist-anomaly]
            [dl4j.facenet]
            [dl4j.paragraph-vec]
            [dl4j.sea-temp]
            [dl4j.instacart]
            [dl4j.clouds]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server :port 7788))