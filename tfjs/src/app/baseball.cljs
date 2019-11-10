(ns app.baseball
  (:require [clojure.string :as string]
            ["@tensorflow/tfjs" :as tf]
            ["fs" :as fs]
            ["path" :as path]
            ["child_process" :as cp]
            ["readline" :as readline]))

#_(path/resolve ".")

(def data-dir (str (path/resolve "./tmp/data/baseball/") "/") )

(defn load-data!
  []
  #_(-> (cp/execSync "ls") (str))
  (-> (cp/execSync "bash data.sh baseball") (str)))

#_(load-data!)

(defn slurp-csv
  [filename]
  (as-> filename x
    (fs/readFileSync x)
    (.toString x)
    (string/split x  #"\n")
    (mapv #(string/split % #",") x)))

#_(do
    (def file (-> (str data-dir "pitch_type_training_data.csv") (fs/readFileSync) (.toString)))
    (def lines (string/split file #"\n"))
    (def header (first lines))
    (def rows (rest lines))
    (def data (->> rows (mapv #(string/split % #","))))
    )

#_(count lines)
#_(count data)
#_(-> rows (last) (string/split #","))
#_(-> header (string/split #","))
#_(->> data (first))

#_(do
    (def csv-training (slurp-csv (str data-dir "pitch_type_training_data.csv")))
    (def csv-test (slurp-csv (str data-dir "pitch_type_test_data.csv"))))

#_(second csv-training)
#_(first csv-test)



