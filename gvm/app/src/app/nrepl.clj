(ns app.nrepl
  (:require [nrepl.server :refer [start-server stop-server]]
            [cider.nrepl :refer [cider-nrepl-handler]]
   ;
            )
  #_(:gen-class))

#_(defn nrepl-handler []
  (require 'cider.nrepl)
  (ns-resolve 'cider.nrepl 'cider-nrepl-handler))

(defn start-nrepl-server [& {:keys [host port] :or {port 7888 host "0.0.0.0"}}]
  (prn "--started nREPL server on 7888 ")
  (start-server
   :bind host
   :port port
   :handler cider-nrepl-handler
   :middleware '[]))

