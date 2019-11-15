(ns el.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [el.core]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server))