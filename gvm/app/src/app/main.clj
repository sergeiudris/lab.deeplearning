(ns app.main
  (:require [clojure.repl :as repl]
            [app.nrepl :refer [start-nrepl-server]])
  (:gen-class))

(defn -main  [& args]
  (prn "hello")
  (start-nrepl-server "0.0.0.0" 7788))