(defproject dl4j "0.1.0"

  :repositories [["central" {:url "https://repo1.maven.org/maven2/"}]
                 ["clojars" {:url "https://clojars.org/repo/"}]
                 ["conjars" {:url "https://conjars.org/repo"}]
                 #_["oracle" {:url "https://download.oracle.com/maven"}]
                 #_["staging" {:url       "https://repository.apache.org/content/repositories/staging"
                               :snapshots true
                               :update    :always}]
                 ["snapshots" {:url       "https://repository.apache.org/content/repositories/snapshots"
                               :snapshots true
                               :update    :always}]]

  :min-lein-version "2.8.0"

  :plugins [[lein-virgil "0.1.9"]]
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [nrepl "0.6.0"]
                 [cider/cider-nrepl "0.22.0"]

                 [org.nd4j/nd4j-cuda-9.2 "1.0.0-beta4"]
                 #_[org.nd4j/nd4j-native "1.0.0-beta4"]]

  :repl-options {:init-ns          dl4j.main
                 :main             dl4j.main
                 :host             "0.0.0.0"
                 :port             7788}
  :profiles {:dev  {:main         ^{:skip-aot true}  dl4j.main
                    :aot          nil #_[dev]
                    :aliases      {"dev" ["trampoline" "run" "-m" "dl4j.main/-main"]}
                    :dependencies []}

             :prod ^:leaky {:main dl4j.main
                            :uberjar-name "dl4j.main-standalone.jar"
                            :jar-name     "dl4j.main.jar"
                            :aot  [dl4j.main]}}


  :main ^{:skip-aot true} dl4j.main
  :jvm-opts ["-Xms768m" "-Xmx11998m"]

  :source-paths ["src"]
  :java-source-paths ["src"]
  :test-paths ["test"]
  :resource-paths ["resources" "config"]
  :auto-clean false)
