(ns tools.user
  (:require [lab.main]))

(defn java-version
  []
  (System/getProperty "java.vm.version"))

(comment


  (System/getProperty "java.vm.version")
  (System/getProperty "java.version")
  (System/getProperty "java.specification.version")
  (clojure-version)

  ;
  )

; issue when failing to compile java classes
; https://www.reddit.com/r/Clojure/comments/5f2ctk/cant_compile_java_sources_with_lein_javac/