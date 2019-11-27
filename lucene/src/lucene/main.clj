(ns lucene.main
  (:require [pad.nrepl.core :refer [start-nrepl-server]]
            [lucene.demo]
   ;
            ))

(defn -main  [& args]
  (start-nrepl-server :port 7788))