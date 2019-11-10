(ns app.baseball
  (:require ["@tensorflow/tfjs" :as tf]
            ["fs" :as fs]
            ["child_process" :as cp]))


(def data-dir "./tmp/data/baseball")

(defn load-data!
  []
  #_(-> (cp/execSync "ls") (str))
  (cp/execSync "bash data.sh baseball"))