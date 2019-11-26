
- gvm
    - Call path from entry point to clojure.spec.gen.alpha$dynaload$fn__2628.invoke():
        - https://github.com/oracle/graal/issues/1266
        - https://github.com/oracle/graal/issues/1681


- dl4j

- issue
    - network issue, host blob.deeplearning4j.org cannot be not resolved
    - http://blob.deeplearning4j.org/datasets/iris.dat doesn't resolve even in Opera with vpn
    - probably a geographical issue
    - resolved
        - https://github.com/eclipse/deeplearning4j-examples/issues/924

- issue
    - > Execution error (DL4JInvalidInputException) at org.deeplearning4j.nn.layers.BaseLayer/preOutputWithPreNorm (BaseLayer.java:306).
        Input that is not a matrix; expected matrix (rank 2), got rank 1 array with shape [784]. Missing preprocessor or wrong input type?
    - https://github.com/eclipse/deeplearning4j/issues/3112
    - resolved
        -
