(ns gvm.polyglot
  (:require [clojure.string :as string])
  (:import
   (org.graalvm.polyglot Context)
   (org.graalvm.polyglot.proxy ProxyArray)))

(comment

  (def ctx (-> (Context/newBuilder (into-array String [ "js"]))
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

  (def m {:id 1
          :text "42"
          :arr [1 2 3]})
  (-> (object-array [m]) (first) (type))
  (def jm (java.util.HashMap. {"id" 1
                               "text" "42"
                               "arr" (java.util.ArrayList. [3 4 5])}))
  (type jm)
  (-> ctx (.getBindings "js") (.putMember "javaObj" jm))
  (def vl (.eval ctx "js" "javaObj"))
  (.hasMembers vl)
  (.getMemberKeys vl)
  (-> ctx (.eval  "js" "javaObj.get('id')") (.asInt))
  (-> ctx (.eval "js" "javaObj.get('arr')[1] == 4") (.asBoolean))

  (def bd (-> ctx
              (.eval  "js" "
                 var BigDecimal = Java.type('java.math.BigDecimal');
                 BigDecimal.valueOf(10).pow(20)
                 ")
              (.asHostObject)))
  (type bd)
  (.toString bd)
  (= (.toString bd) "100000000000000000000")


  (def ar (proxy [ProxyArray] []
            (get [idx] (* idx 2))
            (set [idx vl]
              (throw (UnsupportedOperationException.)))
            (getSize []
              java.lang.Long/MAX_VALUE)))
  (.get ar 3)
  (.getSize ar)
  (.set ar 1 nil)
  (-> ctx (.getBindings "js") (.putMember "arr" ar))
  (def rlt (-> ctx
               (.eval "js" "
                      arr[1] + arr[1000000000]
                      ")
               (.asLong)))
  (type rlt)
  (== (long 2000000002) rlt )

  (.close ctx)

  ;
  )

; https://www.graalvm.org/docs/reference-manual/languages/python/
(comment

  (def ctx (-> (Context/newBuilder (into-array String ["python"]))
               (.allowAllAccess true)
               (.build)))

  (-> ctx (.getEngine) (.getLanguages) (.keySet) (set))

  (-> ctx (.eval "python" "2 * 2") (.asInt))

  (-> ctx (.eval "python" (str
                           "import time;"
                           "time.time();")))

  (def pytime (-> ctx (.eval "python" (str
                                       "import time;"
                                       "time.time;"))))
  (.canExecute pytime)
  (.execute pytime (object-array []))


  ;
  )