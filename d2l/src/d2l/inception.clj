(ns d2l.inception
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.visualization :as viz]))

(def data-dir "./tmp/data/inception/")

(defn load-data!
  []
  (when-not  (.exists (io/file (str data-dir "Inception-BN-symbol.json")))
    (do
      (:exit (sh "bash" "-c" "bash bin/data.sh inception" :dir "/opt/app")))))

#_(load-data!)



(def h 224) ;; Image height
(def w 224) ;; Image width
(def c 3)   ;; Number of channels: Red, Green, Blue

(comment

  ;; Loading VGG16
  (defonce vgg-16-mod
    (-> {:prefix (str data-dir "vgg16") :epoch 0}
        (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
        (m/bind {:for-training false
                 :data-shapes [{:name "data" :shape [1 c h w]}]})))

;; Loading Inception v3
  (defonce inception-mod
    (-> {:prefix (str data-dir "Inception-BN") :epoch 0}
        (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
        (m/bind {:for-training false
                 :data-shapes [{:name "data" :shape [1 c h w]}]})))


  ;
  )

