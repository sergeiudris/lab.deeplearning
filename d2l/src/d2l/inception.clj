(ns d2l.inception
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.visualization :as viz]

            [opencv4.colors.rgb :as rgb]
            [opencv4.mxnet :as mx-cv]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

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

(defn preprocess-img-mat
  "Preprocessing steps on an `img-mat` from OpenCV to feed into the Model"
  [img-mat]
  (-> img-mat
      ;; Resize image to (w, h)
      (cv/resize! (cv/new-size w h))
      ;; Maps pixel values from [-128, 128] to [0, 127]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; Substract mean pixel values from ImageNet dataset
      (cv/add! (cv/new-scalar -103.939 -116.779 -123.68))
      ;; Flatten matrix
      (cvu/mat->flat-rgb-array)
      ;; Reshape to (1, c, h, w)
      (ndarray/array [1 c h w])))

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


  (defonce image-net-labels
    (-> (str data-dir "/synset.txt")
        (slurp)
        (str/split #"\n")))

  ;; ImageNet 1000 Labels check
  (assert (= 1000 (count image-net-labels)))

  ;
  )

