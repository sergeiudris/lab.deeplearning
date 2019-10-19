(ns app.main
  (:require [tools.nrepl]
            [app.core]
            [examples.translation.core]
   ;
            ))

#_(examples.translation.core/ping)

(defn -main  [& args]
  (tools.nrepl/-main))