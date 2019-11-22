(ns app.main
  (:require [clojure.repl :as repl]
            #_[pad.nrepl.core :refer [start-nrepl-server]]
            [pad.core :refer [java-version memory-info]])
  (:gen-class))

(defn -main  [& args]
  (prn "hello")
  #_(start-nrepl-server)
  (prn (java-version))
  (prn (memory-info)))