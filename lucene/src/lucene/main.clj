(ns lucene.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server :port 7788))