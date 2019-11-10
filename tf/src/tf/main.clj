(ns tf.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [tf.core]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server))