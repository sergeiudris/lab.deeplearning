(ns app.main
  (:require [clojure.repl :as repl]
            [app.nrepl :refer [start-nrepl-server]]
            [pad.core :refer [java-version memory-info]])
  (:gen-class))

(defn -main  [& args]
  (prn "hello")
  (start-nrepl-server)
  (prn (java-version))
  (prn (memory-info)))