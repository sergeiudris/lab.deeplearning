(ns lucene.demo
  (:require [clojure.string :as string])
  (:import
   org.apache.lucene.demo.IndexFiles
   org.apache.lucene.demo.SearchFiles))

(comment

  (IndexFiles/main (into-array ["-docs" "/opt/app/src/"
                                "-index" "/opt/app/tmp/index/"]))

  (SearchFiles/main (into-array ["-index" "/opt/app/tmp/index/"
                                 "-query" "into-array"
                                 ]))
  
  (SearchFiles/query "/opt/app/tmp/index/" "into-array" )

  ;
  )