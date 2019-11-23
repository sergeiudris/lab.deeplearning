(ns gvm.polyglot
  (:require [clojure.string :as string])
  (:import
   (org.graalvm.polyglot Context)))

(comment

  (def ctx (-> (Context/newBuilder (into-array String ["js"]))
               (.allowAllAccess true)
               (.build)))

  (def langs (-> ctx (.getEngine) (.getLanguages) (.keySet) (set) ))
  (type langs)

  (.eval ctx "js" "print('Hello world!')")
  (.close ctx)

  ;
  )