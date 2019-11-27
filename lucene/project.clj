(defproject dl4j "0.1.0"

  :repositories [["central" {:url "https://repo1.maven.org/maven2/"}]
                 ["clojars" {:url "https://clojars.org/repo/"}]
                 ["conjars" {:url "https://conjars.org/repo"}]
                 ["maven-restlet" {:url "https://maven.restlet.com/"}]
                 #_["oracle" {:url "https://download.oracle.com/maven"}]
                 #_["staging" {:url       "https://repository.apache.org/content/repositories/staging"
                               :snapshots true
                               :update    :always}]
                 #_["snapshots" {:url       "https://repository.apache.org/content/repositories/snapshots"
                                 :snapshots true
                                 :update    :always}]]

  :min-lein-version "2.8.0"

  :plugins [[lein-virgil "0.1.9"]]
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [nrepl "0.6.0"]
                 [cider/cider-nrepl "0.22.0"]
                 [org.clojure/data.json "0.2.7"]
                 [commons-io/commons-io "2.5"]

                 [org.apache.lucene/lucene-core "8.3.0"]
                 [org.apache.lucene/lucene-facet "8.3.0"]
                 [org.apache.lucene/lucene-analyzers-common "8.3.0"]
                 [org.apache.lucene/lucene-queryparser "8.3.0"]
                 ]


  :repl-options {:init-ns          lucene.main
                 :main             lucene.main
                 :host             "0.0.0.0"
                 :port             7788}
  :profiles {:dev  {:main         ^{:skip-aot true}  lucene.main
                    :aot          nil #_[dev]
                    :aliases      {"dev" ["trampoline" "run" "-m" "lucene.main/-main"]}
                    :dependencies []}

             :prod ^:leaky {:main lucene.main
                            :uberjar-name "lucene.main-standalone.jar"
                            :jar-name     "lucene.main.jar"
                            :aot  [lucene.main]}}


  :main ^{:skip-aot true} lucene.main
  :jvm-opts ["-Xms768m" "-Xmx11998m"]

  :source-paths ["src"]
  :java-source-paths ["src"]
  :test-paths ["test"]
  :resource-paths ["resources" "config"]
  :auto-clean false)
