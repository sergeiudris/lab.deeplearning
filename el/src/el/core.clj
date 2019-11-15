(ns el.core
  (:require [clojure.pprint :as pp]
            [pad.prn.core :refer [linst]]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.ndarray :as nd]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.shape :as shape]))