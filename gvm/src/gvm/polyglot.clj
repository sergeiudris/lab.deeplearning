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
  
  (def f (.eval ctx "js" "(x)=> x + 1"))
  (.canExecute f)
  (type (. clojure.lang.RT object_array [1 "2"]))
  (def vl (.execute f (object-array [41])))
  (.asInt vl)
  (type (.asInt vl))
  (.close ctx)

  ;
  )