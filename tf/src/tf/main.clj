(ns tf.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [tf.core]
            [tf.tensorflow]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server))