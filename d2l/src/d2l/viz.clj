(ns d2l.viz
  (:require [clojure.java.io :as io]
            [clojure.java.shell :refer [sh]]
            [clojure.string :as str]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.visualization :as viz]))

(def data-dir "./tmp/data/viz/")

(defn load-data!
  []
  (when-not  (.exists (io/file (str data-dir "vgg16-symbol.json")))
    (do
      (:exit (sh "bash" "-c" "bash bin/data.sh viz" :dir "/opt/app")))))

#_(load-data!)

(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
             model-sym
             {"data" input-data-shape}
             {:title model-name
              :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot model-name path)))

(comment

  (def vgg16-mod
    "VGG16 Module"
    (m/load-checkpoint {:prefix (str data-dir "/vgg16") :epoch 0}))

  (def resnet18-mod
    "Resnet18 Module"
    (m/load-checkpoint {:prefix (str data-dir "/resnet-18") :epoch 0}))


  (render-model! {:model-name "vgg16"
                  :model-sym (m/symbol vgg16-mod)
                  :input-data-shape [1 3 244 244]
                  :path data-dir})

  (render-model! {:model-name "resnet18"
                  :model-sym (m/symbol resnet18-mod)
                  :input-data-shape [1 3 244 244]
                  :path data-dir})
  

  ;
  )
