(ns app.baseball
  (:require [clojure.string :as string]
            ["@tensorflow/tfjs-node-gpu" :as tf]
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

(defn normalize
  [v vmin vmax]
  (if (or (not vmin) (not vmax))
    v
    (/ (- v vmin) (- vmax vmin))))

(def TRAIN_DATA_PATH (str "file://" data-dir "pitch_type_training_data.csv"))
(def TEST_DATA_PATH (str "file://" data-dir "pitch_type_test_data.csv"))

(def VX0_MIN -18.885)
(def VX0_MAX 18.065)
(def VY0_MIN -152.463)
(def VY0_MAX -86.374)
(def VZ0_MIN -15.5146078412997)
(def VZ0_MAX 9.974)
(def AX_MIN -48.0287647107959)
(def AX_MAX 30.592)
(def AY_MIN 9.397)
(def AY_MAX 49.18)
(def AZ_MIN -49.339)
(def AZ_MAX 2.95522851438373)
(def START_SPEED_MIN 59)
(def START_SPEED_MAX 104.4)
(def NUM_PITCH_CLASSES 7)
(def TRAINING_DATA_LENGTH 7000)
(def TEST_DATA_LENGTH 700)

(defn csv-transform
  [opts]
  (let [xs (aget opts "xs")
        ys (aget opts "ys")
        values #js [(normalize (aget xs "vx0") VX0_MIN VX0_MAX)
                    (normalize (aget xs "vy0") VY0_MIN VY0_MAX)
                    (normalize (aget xs "vz0") VZ0_MIN VZ0_MAX)
                    (normalize (aget xs "ax") AX_MIN AX_MAX)
                    (normalize (aget xs "ay") AY_MIN AY_MAX)
                    (normalize (aget xs "az") AZ_MIN AZ_MAX)
                    (normalize (aget xs "start_speed") START_SPEED_MIN START_SPEED_MAX)
                    (aget xs "left_handed_pitcher")]]
    #js {"xs" values
         "ys" (aget ys "pitch_code")}))

(defn create-model
  []
  (let [model (tf/sequential)]
    (do
      (.add model (tf/layers.dense (clj->js {:units 250 :activation "relu" :inputShape [8]})))
      (.add model (tf/layers.dense (clj->js {:units 175 :activation "relu"})))
      (.add model (tf/layers.dense (clj->js {:units 150 :activation "relu"})))
      (.add model (tf/layers.dense (clj->js {:units NUM_PITCH_CLASSES :activation "softmax"})))
      (.compile model (clj->js {:optimizer (tf/train.adam)
                                :loss "sparseCategoricalCrossentropy"
                                :metrics ["accuracy"]})))
    model
    ))



(defn class-num>>pitch
  [class-num]
  (case class-num
    0 "Fastball (2-seam)"
    1 "Fastball (4-seam)"
    2 "Fastball (sinker)"
    3 "Fastball (cutter)"
    4 "Slider"
    5 "Changeup"
    6 "Curveball"
    "Unknown"))

(defn calc-pitch-class-eval
  [pitch-idx class-size vals]
  (let [idx (+ (* pitch-idx class-size NUM_PITCH_CLASSES) pitch-idx)]
    (/
     (reduce (fn [a i]
               (+ a (aget vals (+ idx (* i NUM_PITCH_CLASSES)))))
             0 (range 0 class-size))
     class-size)
    ))

(defn evaluate-data
  [{:keys [model data data-count num-classes assoc-key reduce-init-val]}]
  (js/Promise.
   (fn [resolve reject]
     (.forEachAsync data
                    (fn [pitch-type-batch]
                      (let [vals (-> model
                                     (.predict (aget pitch-type-batch "xs"))
                                     (.dataSync))
                            class-size (/ data-count num-classes)
                            results (reduce (fn [a i]
                                              (assoc-in a [(class-num>>pitch i) assoc-key]
                                                        (calc-pitch-class-eval i class-size vals)))
                                            reduce-init-val (range 0 num-classes))]
                        (resolve results)))))))

(defn evaluate
  [model train-valid-data test-valid-data user-test-data?]
  (let [results-p (evaluate-data {:model model
                                  :data train-valid-data
                                  :data-count TRAINING_DATA_LENGTH
                                  :num-classes NUM_PITCH_CLASSES
                                  :assoc-key :training
                                  :reduce-init-val {}})
        results-p2 (when user-test-data?
                     (-> results-p
                         (.then (fn [results]
                                  (evaluate-data {:model model
                                                  :data test-valid-data
                                                  :data-count TEST_DATA_LENGTH
                                                  :num-classes NUM_PITCH_CLASSES
                                                  :assoc-key :validation
                                                  :reduce-init-val results})))))]
    (or results-p2 results-p)))

(defn predict-sample
  [model sample]
  (let [result (-> model
                    (.predict (tf/tensor (clj->js sample) #js [1 (count sample) ]))
                    (.arraySync))]
    (->>
     (reduce (fn [a i]
               (if (> (aget result 0 i) (:vmax a))
                 (merge a {:predicted-pitch-idx i
                           :vmax (aget result 0 i)})
                 a
                 ))
             {:vmax 0
              :predicted-pitch-idx 7}
             (range 0 NUM_PITCH_CLASSES))
     :predicted-pitch-idx
     (class-num>>pitch))))

(def TIMEOUT_BETWEEN_EPOCHS_MS 500)
(def NUM_EPOCHS 10)

(comment

  (def train-data
    (-> TRAIN_DATA_PATH
        (tf/data.csv (clj->js {"columnConfigs" {"pitch_code" {"isLabel" true}}}))
        (.map csv-transform)
        (.shuffle TRAINING_DATA_LENGTH)
        (.batch 100)))

  (def train-valid-data
    (-> TRAIN_DATA_PATH
        (tf/data.csv (clj->js {"columnConfigs" {"pitch_code" {"isLabel" true}}}))
        (.map csv-transform)
        (.batch TRAINING_DATA_LENGTH)))

  (def test-valid-data
    (-> TEST_DATA_PATH
        (tf/data.csv (clj->js {"columnConfigs" {"pitch_code" {"isLabel" true}}}))
        (.map csv-transform)
        (.batch TEST_DATA_LENGTH)))

  (def model (create-model))

  (def results$ (atom nil))
  
  (do @results$)
  
  (-> model
      (.fitDataset  train-data #js {:epochs 10})
      (.then (fn [his]
               (evaluate model train-valid-data test-valid-data true)
               ))
      (.then (fn [results]
               (prn "--finished, reset results atom")
               (reset! results$ results))
             ))
  
  (def curveball [2.668 -114.333 -1.908 4.786 25.707 -45.21 78 0])
  
  (predict-sample model curveball)
  ; "Fastball (2-seam)"
  
  (predict-sample model [2.668 -154.333 -1.908 4.786 25.707 -45.21 78 0])
  ; "Fastball (4-seam)"
  
  
  )
