An Information Theoretic Feature Selection Framework
=====================================================

The present framework implements Feature Selection (FS) on Flink for its application on Big Data problems. This package contains a generic implementation of greedy Information Theoretic Feature Selection methods. The implementation is based on the common theoretic framework presented in [1]. Implementations of mRMR, InfoGain, JMI and other commonly used FS filters are provided. In addition, the framework can be extended with other criteria provided by the user as long as the process complies with the framework proposed in [1].

## Incoming features:

* Support for sparse data and high-dimensional datasets (in progress).

## Example: 
    import org.apache.flink.ml.preprocessing.InfoSelector
  	val nfeat = 10 // Number of features to be selected
    val ni = 100000 // Number of instances
    val nf = 100 // Number of features
    val selector = InfoSelector().setNFeatures(nfeat).setNF(nf).setNI(ni)
    selector.fit(training)
    println("Selected features: " + selector.selectedFeatures.get.mkString(","))


## Prerequisites:

Data must be discretized as integer values in double representation with a maximum of 256 distinct values. 
By doing so, data can be transformed to byte type directly, making the selection process much more efficient.

## Contributors

- Sergio Ramírez-Gallego (sramirez@decsai.ugr.es) (main contributor and maintainer)

##References

[1] Brown, G., Pocock, A., Zhao, M. J., & Luján, M. (2012). "Conditional likelihood maximisation: a unifying framework for information theoretic feature selection." The Journal of Machine Learning Research, 13(1), 27-66.
