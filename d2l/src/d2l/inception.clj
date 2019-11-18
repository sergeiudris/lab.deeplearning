(ns d2l.inception
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as nd]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.visualization :as viz]
            [opencv4.colors.rgb :as rgb]
            [opencv4.mxnet :as mx-cv]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

(def opts
  {:dir/shell "/opt/app/"
   :dir/target "/opt/app/tmp/data/inception/"})

(defn data-dir
  [{:dir/keys [target]}]
  target)

(defn script-fetch-inception
  [{:dir/keys [target]}]
  (format "
  DIR=%s
  mkdir -p $DIR
  cd $DIR

  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params

  wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-symbol.json
  wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-0126.params
  mv Inception-BN-0126.params Inception-BN-0000.params

  wget http://data.mxnet.io/models/imagenet/synset.txt

  # images
  wget https://arthurcaillau.com/assets/images/cat-egyptian.jpg
  wget https://arthurcaillau.com/assets/images/dog-2.jpg
  wget https://arthurcaillau.com/assets/images/guitarplayer.jpg
  " target))

(defn fetch-inception
  [{:dir/keys [shell] :as opts}]
  (sh "bash" "-c" (script-fetch-inception opts) :dir shell))

#_(.exists (io/file (str (data-dir opts) "Inception-BN-symbol.json")))
#_(:exit (sh "bash" "-c" (format "rm -rf %s" (data-dir opts)) :dir (opts :dir/shell)))
#_(:exit (fetch-inception opts))

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
      (nd/array [1 c h w])))

(defn- top-k
  "Return top `k` from prob-maps with :prob key"
  [k prob-maps]
  (->> prob-maps
       (sort-by :prob)
       (reverse)
       (take k)))

(defn predict
  "Predict with `model` the top `k` labels from `labels` of the ndarray `x`"
  ([model labels x]
   (predict model labels x 5))
  ([model labels x k]
   (let [probs (-> model
                   (m/forward {:data [x]})
                   (m/outputs)
                   (ffirst)
                   (nd/->vec))
         prob-maps (mapv (fn [p l] {:prob p :label l}) probs labels)]
     (top-k k prob-maps))))

(comment

  (defonce image-net-labels
    (-> (str (data-dir opts) "/synset.txt")
        (slurp)
        (str/split #"\n")))

  ;; ImageNet 1000 Labels check
  (assert (= 1000 (count image-net-labels)))

  ;; Loading VGG16
  (defonce vgg-16-mod
    (-> {:prefix (str (data-dir opts) "vgg16") :epoch 0 :contexts [(context/gpu 0)]}
        (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
        (m/bind {:for-training false
                 :data-shapes [{:name "data" :shape [1 c h w]}]})))

  ;; Loading Inception v3
  (defonce inception-mod
    (-> {:prefix (str (data-dir opts) "Inception-BN") :epoch 0 :contexts [(context/gpu 0)]}
        (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
        (m/bind {:for-training false
                 :data-shapes [{:name "data" :shape [1 c h w]}]})))

  (->> (str (data-dir opts) "cat-egyptian.jpg")
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))

  ; ({:label "n02124075 Egyptian cat", :prob 0.9669786}
  ;  {:label "n02123045 tabby, tabby cat", :prob 0.020066934}
  ;  {:label "n02123159 tiger cat", :prob 0.0071042604}
  ;  {:label "n02127052 lynx, catamount", :prob 0.005353977}
  ;  {:label "n02123597 Siamese cat, Siamese", :prob 4.658181E-5})

  (->> (str (data-dir opts) "cat-egyptian.jpg")
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))

  ; ({:label "n02124075 Egyptian cat", :prob 0.9030159}
  ;  {:label "n02123045 tabby, tabby cat", :prob 0.05147691}
  ;  {:label "n02123159 tiger cat", :prob 0.024212532}
  ;  {:label "n02127052 lynx, catamount", :prob 0.009907055}
  ;  {:label "n04040759 radiator", :prob 3.7205184E-4})




  (->> (str (data-dir opts) "dog-2.jpg")
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))

  ; ({:label "n02110958 pug, pug-dog", :prob 0.73638505}
  ;  {:label "n02108422 bull mastiff", :prob 0.23988225}
  ;  {:label "n02108915 French bulldog", :prob 0.013495391}
  ;  {:label "n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier"
  ;   :prob 0.0019004609}
  ;  {:label "n04409515 tennis ball", :prob 0.0013417462})

  (->> (str (data-dir opts) "dog-2.jpg")
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))

    ; ({:label "n02110958 pug, pug-dog", :prob 0.9562809}
    ;  {:label "n02108422 bull mastiff", :prob 0.022715751}
    ;  {:label "n02108915 French bulldog", :prob 0.007526083}
    ;  {:label "n02086079 Pekinese, Pekingese, Peke", :prob 0.001468682}
    ;  {:label "n02108089 boxer", :prob 0.0012910493})




  (->> (str (data-dir opts) "guitarplayer.jpg")
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))

  ; ({:label "n03272010 electric guitar", :prob 0.64720076}
  ;  {:label "n04296562 stage", :prob 0.3371945}
  ;  {:label "n02676566 acoustic guitar", :prob 0.00880973}
  ;  {:label "n02787622 banjo", :prob 0.0024602269}
  ;  {:label "n03759954 microphone, mike", :prob 0.0018765893})

  (->> (str (data-dir opts) "guitarplayer.jpg")
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))

  ; ({:label "n03272010 electric guitar", :prob 0.73966444}
  ;  {:label "n04296562 stage", :prob 0.105859876}
  ;  {:label "n04141076 sax, saxophone", :prob 0.059584014}
  ;  {:label "n02787622 banjo", :prob 0.029627405}
  ;  {:label "n02676566 acoustic guitar", :prob 0.016049426})

  ;
  )

