(ns gvm.polyglot
  (:require [clojure.string :as string])
  (:import
   (org.graalvm.polyglot Context)))

(comment

  (def ctx (-> (Context/newBuilder (into-array String ["js"]))
               (.allowAllAccess true)
               (.build)))

  (def langs (-> ctx (.getEngine) (.getLanguages) (.keySet) (set)))
  (type langs)

  (.eval ctx "js" "print('Hello world!')")

  (def f (.eval ctx "js" "(x)=> x + 1"))
  (.canExecute f)
  (type (. clojure.lang.RT object_array [1 "2"]))
  (def vl (.execute f (object-array [41])))
  (.asInt vl)
  (type (.asInt vl))

  (def ob (.eval ctx "js" "
                 ({  
                  id : 1,
                  text : '42',
                  arr : [3,4,5]
                 })
                 "))
  (.hasMembers ob)
  (.getMemberKeys ob)
  (-> ob (.getMember "id") (.asInt))
  (-> ob (.getMember "text") (.asString) (type))
  (def ar (-> ob (.getMember "arr")))
  (.hasArrayElements ar)
  (.getArraySize ar)
  (-> ar (.getArrayElement 1) (.asInt))

  (.close ctx)

  ;
  )