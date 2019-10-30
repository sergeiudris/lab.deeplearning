(ns d2l.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [d2l.core]
            [d2l.mnist]
            [d2l.house]
            [d2l.mxnet-made-simple]
            [d2l.viz]
            
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server))