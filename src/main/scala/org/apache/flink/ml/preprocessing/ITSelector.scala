/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.preprocessing

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM}
import breeze.numerics.sqrt
import breeze.numerics.sqrt._
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.flink.ml.preprocessing.{InfoThCriterionFactory => FT}
import org.apache.flink.ml.common.{LabeledVector, Parameter, ParameterMap}
import org.apache.flink.ml.math.Breeze._
import org.apache.flink.ml.math.{BreezeVectorConverter, Vector, DenseVector, SparseVector}
import org.apache.flink.ml.pipeline.{TransformOperation, FitOperation,
Transformer}
import org.apache.flink.ml.preprocessing.ITSelector._
import scala.reflect.ClassTag

/** Scales observations, so that all features have a user-specified mean and standard deviation.
  * By default for [[StandardScaler]] transformer mean=0.0 and std=1.0.
  *
  * This transformer takes a subtype of  [[Vector]] of values and maps it to a
  * scaled subtype of [[Vector]] such that each feature has a user-specified mean and standard
  * deviation.
  *
  * This transformer can be prepended to all [[Transformer]] and
  * [[org.apache.flink.ml.pipeline.Predictor]] implementations which expect as input a subtype
  * of [[Vector]].
  *
  * @example
  *          {{{
  *            val trainingDS: DataSet[Vector] = env.fromCollection(data)
  *            val transformer = StandardScaler().setMean(10.0).setStd(2.0)
  *
  *            transformer.fit(trainingDS)
  *            val transformedDS = transformer.transform(trainingDS)
  *          }}}
  *
  * =Parameters=
  *
  * - [[Mean]]: The mean value of transformed data set; by default equal to 0
  * - [[Std]]: The standard deviation of the transformed data set; by default
  * equal to 1
  */
class ITSelector extends Transformer[ITSelector] {

  private[preprocessing] var selectedFeatures: Option[Array[Int]] = None

  /** Sets the target mean of the transformed data
    *
    * @param mu the user-specified mean value.
    * @return the StandardScaler instance with its mean value set to the user-specified value
    */
  def setNToSelect(n: Int): ITSelector = {
    parameters.add(NToSelect, n)
    this
  }
  
  def setNPartitions(np: Int): ITSelector = {
    parameters.add(NPartitions, np)
    this
  }
    
  def setCriterion(c: String): ITSelector = {
    parameters.add(Criterion, c)
    this
  }
}

object ITSelector {

  // ====================================== Parameters =============================================

  case object NToSelect extends Parameter[Int] {
    override val defaultValue: Option[Int] = Some(10)
  }

  case object NPartitions extends Parameter[Int] {
    override val defaultValue: Option[Int] = Some(0)
  }
  
  case object Criterion extends Parameter[String] {
    override val defaultValue: Option[String] = Some("mrmr")
  }
  

  // ==================================== Factory methods ==========================================

  def apply(): ITSelector = {
    new ITSelector()
  }

  // ====================================== Operations =============================================

  /** Trains the [[StandardScaler]] by learning the mean and standard deviation of the training
    * data which is of type [[LabeledVector]]. The mean and standard deviation are used to
    * transform the given input data.
    *
    */
  implicit val fitLabeledVectorStandardScaler = {
    new FitOperation[ITSelector, LabeledVector] {
      override def fit(
          instance: ITSelector,
          fitParameters: ParameterMap,
          input: DataSet[LabeledVector])
        : Unit = {
        val metrics = extractFeatures(input, fitParameters)
        instance.selectedFeatures = Some(metrics)
      }
    }
  }

  /** Trains the [[StandardScaler]] by learning the mean and standard deviation of the training
    * data which is of type ([[Vector]], Double). The mean and standard deviation are used to
    * transform the given input data.
    *
    */
  /*implicit def fitLabelVectorTupleStandardScaler
  [T <: Vector: BreezeVectorConverter: TypeInformation: ClassTag] = {
    new FitOperation[ITSelector, (T, Double)] {
      override def fit(
          instance: ITSelector,
          fitParameters: ParameterMap,
          input: DataSet[(T, Double)])
      : Unit = {
        val vectorDS = input.map(_._1)
        val metrics = extractFeatureMetrics(vectorDS)

        instance.metricsOption = Some(metrics)
      }
    }
  }*/
  
  
  // Case class for criteria/feature
  protected case class F(feat: Int, crit: Double) 
  // Case class for columnar data (dense and sparse version)
  private case class ColumnarData(dense: DataSet[(Int, Array[Byte])], 
      sparse: DataSet[(Int, BV[Byte])],
      isDense: Boolean,
      originalNPart: Int)

  /**
   * Performs a info-theory FS process.
   * 
   * @param data Columnar data (last element is the class attribute).
   * @param nInstances Number of samples.
   * @param nFeatures Number of features.
   * @return A list with the most relevant features and its scores.
   * 
   */
  private[preprocessing] def selectFeatures(
      data: ColumnarData,
      params: ParameterMap,
      nInstances: Long,
      nFeatures: Int) = {
    
    val label = nFeatures - 1
    // Initialize all criteria with the relevance computed in this phase. 
    // It also computes and saved some information to be re-used.
    val (it, relevances) = if(data.isDense) {
      val it = InfoTheory.initializeDense(data.dense, label, nInstances, nFeatures, data.originalNPart)
      (it, it.relevances)
    } else {
      val it = InfoTheory.initializeSparse(data.sparse, label, nInstances, nFeatures)
      (it, it.relevances)
    }

    // Initialize all (except the class) criteria with the relevance values
    val cFactory = new InfoThCriterionFactory(params(Criterion))
    val pool = Array.fill[InfoThCriterion](nFeatures - 1) {
      val crit = cFactory.getCriterion.init(Float.NegativeInfinity)
      crit.setValid(false)
    }    
    relevances.collect().foreach{ case (x, mi) => 
      pool(x) = cFactory.getCriterion.init(mi.toFloat) 
    }
    
    // Print most relevant features
    val topByRelevance = relevances.sortBy(_._2, false).take(nToSelect)
    val strRels = topByRelevance.map({case (f, mi) => (f + 1) + "\t" + "%.4f" format mi})
      .mkString("\n")
    println("\n*** MaxRel features ***\nFeature\tScore\n" + strRels) 
    
    // Get the maximum and initialize the set of selected features with it
    val (max, mid) = pool.zipWithIndex.maxBy(_._1.relevance)
    var selected = Seq(F(mid, max.score))
    pool(mid).setValid(false)
      
    // MIM does not use redundancy, so for this criterion all the features are selected now
    if (criterionFactory.getCriterion.toString == "MIM") {
      selected = topByRelevance.map({case (id, relv) => F(id, relv)}).reverse
    }
    
    var moreFeat = true
    // Iterative process for redundancy and conditional redundancy
    while (selected.size < nToSelect && moreFeat) {

      val redundancies = it match {
        case dit: InfoTheoryDense => dit.getRedundancies(selected.head.feat)
        case sit: InfoTheorySparse => sit.getRedundancies(selected.head.feat)
      }
      
      // Update criteria with the new redundancy values
      redundancies.collect().par.foreach({case (k, (mi, cmi)) =>
         pool(k).update(mi.toFloat, cmi.toFloat) 
      })
      
      // select the best feature and remove from the whole set of features
      val (max, maxi) = pool.par.zipWithIndex.filter(_._1.valid).maxBy(_._1)
      if(maxi != -1){
        selected = F(maxi, max.score) +: selected
        pool(maxi).setValid(false)
      } else {
        moreFeat = false
      }
    }
    selected.reverse
  }

  /**
   * Process in charge of transforming data in a columnar format and launching the FS process.
   * 
   * @param data RDD of LabeledPoint.
   * @return A feature selection model which contains a subset of selected features.
   * 
   */
  def extractFeatures(data: DataSet[LabeledVector], 
          fitParameters: ParameterMap): Array[Int] = {  
    
    /*if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    
    if(data.mapPartitions(it => Seq(it.size).toIterator).distinct().count() > 1) {
      logError("The dataset must be split in equal-sized partitions.")
    }*/
      
    // Feature vector must be composed of bytes, not the class
    val requireByteValues = (v: Vector) => {        
      val values = v match {
        case sv: SparseVector =>
          sv.data
        case dv: DenseVector =>
          dv.data
      }
      val condition = (value: Double) => value <= Byte.MaxValue && 
        value >= Byte.MinValue && value % 1 == 0.0
      if (!values.forall(condition(_))) {
        throw new Exception(
            s"Info-Theoretic Framework requires positive values in range [0, 255]")
      }           
    }
        
    // Get basic info
    val first = data.first(1).collect()(0)
    val dense = first.vector.isInstanceOf[DenseVector]    
    val nInstances = data.count()
    val nFeatures = first.vector.size + 1
    require(fitParameters(NToSelect) < nFeatures)  
    
    // Start the transformation to the columnar format
    val colData = if(dense) {
      
      val np = if(fitParameters(NPartitions) == 0) nFeatures else fitParameters(NPartitions)
      /*if(np > nFeatures) {
        logWarning("Number of partitions should be equal or less than the number of features."
          + " At least, less than 2x the number of features.")
      }*/
      
      val classMap = data.map(_.label).distinct.collect()
        .zipWithIndex.map(t => t._1 -> t._2.toByte)
        .toMap
      
      // Transform data into a columnar format by transposing the local matrix in each partition
      val columnarData = data.mapPartitionsWithIndex({ (index, it) =>
        val data = it.toArray
        val mat = Array.ofDim[Byte](nFeatures, data.length)
        var j = 0
        for(reg <- data) {
          requireByteValues(reg.features)
          for(i <- 0 until reg.features.size) mat(i)(j) = reg.features(i).toByte
          mat(reg.features.size)(j) = classMap(reg.label)
          j += 1
        }
        
        val chunks = for(i <- 0 until nFeatures) yield ((i * numPartitions + index) -> mat(i))
        chunks.toIterator
      })      
      
      // Sort to group all chunks for the same feature closely. 
      // It will avoid to shuffle too much histograms
      val denseData = columnarData.sortByKey(numPartitions = np).persist(StorageLevel.MEMORY_ONLY)
      
      ColumnarData(denseData, null, true, data.partitions.size)      
    } else {      
      
      val np = if(numPartitions == 0) data.conf.getInt("spark.default.parallelism", 750) else numPartitions
      val classMap = data.map(_.label).distinct.collect()
        .zipWithIndex.map(t => t._1 -> t._2.toByte)
        .toMap
        
      val sparseData = data.zipWithIndex().flatMap ({ case (lp, r) => 
          requireByteValues(lp.features)
          val sv = lp.features.asInstanceOf[SparseVector]
          val output = (nFeatures - 1) -> (r, classMap(lp.label))
          val inputs = for(i <- 0 until sv.indices.length) 
            yield (sv.indices(i), (r, sv.values(i).toByte))
          output +: inputs           
      })
      
      // Transform sparse data into a columnar format 
      // by grouping all values for the same feature in a single vector
      val columnarData = sparseData.groupByKey(new HashPartitioner(np))
        .mapValues({a => 
          if(a.size >= nInstances) {
            val init = Array.fill[Byte](nInstances.toInt)(0)
            val result: BV[Byte] = new BDV(init)
            a.foreach({case (k, v) => result(k.toInt) = v})
            result
          } else {
            val init = a.toArray.sortBy(_._1)
            new BSV(init.map(_._1.toInt), init.map(_._2), nInstances.toInt)
          }
        }).persist(StorageLevel.MEMORY_ONLY)
      
      ColumnarData(null, columnarData, false, data.partitions.size)
    }
    
    // Start the main algorithm
    val selected = selectFeatures(colData, nInstances, nFeatures)          
    if(dense) colData.dense.unpersist() else colData.sparse.unpersist()
  
    // Print best features according to the mRMR measure
    val out = selected.map{case F(feat, rel) => 
        (feat + 1) + "\t" + "%.4f".format(rel)
      }.mkString("\n")
    logInfo("\n*** Selected features ***\nFeature\tScore\n" + out)
    // Features must be sorted
    //new InfoThSelectorModel(selected.map{case F(feat, rel) => feat}.sorted.toArray)
    selected.map{case F(feat, rel) => feat}.sorted.toArray
  }

  /** Calculates in one pass over the data the features' mean and standard deviation.
    * For the calculation of the Standard deviation with one pass over the data,
    * the Youngs & Cramer algorithm was used:
    * [[http://www.cs.yale.edu/publications/techreports/tr222.pdf]]
    *
    *
    * @param dataSet The data set for which we want to calculate mean and variance
    * @return  DataSet containing a single tuple of two vectors (meanVector, stdVector).
    *          The first vector represents the mean vector and the second is the standard
    *          deviation vector.
    */
  private def extractFeatureMetrics[T <: Vector](dataSet: DataSet[T])
  : DataSet[(linalg.Vector[Double], linalg.Vector[Double])] = {
    val metrics = dataSet.map{
      v => (1.0, v.asBreeze, linalg.Vector.zeros[Double](v.size))
    }.reduce{
      (metrics1, metrics2) => {
        /* We use formula 1.5b of the cited technical report for the combination of partial
           * sum of squares. According to 1.5b:
           * val temp1 : m/n(m+n)
           * val temp2 : n/m
           */
        val temp1 = metrics1._1 / (metrics2._1 * (metrics1._1 + metrics2._1))
        val temp2 = metrics2._1 / metrics1._1
        val tempVector = (metrics1._2 * temp2) - metrics2._2
        val tempS = (metrics1._3 + metrics2._3) + (tempVector :* tempVector) * temp1

        (metrics1._1 + metrics2._1, metrics1._2 + metrics2._2, tempS)
      }
    }.map{
      metric => {
        val varianceVector = sqrt(metric._3 / metric._1)

        for (i <- 0 until varianceVector.size) {
          if (varianceVector(i) == 0.0) {
            varianceVector.update(i, 1.0)
          }
        }
        (metric._2 / metric._1, varianceVector)
      }
    }
    metrics
  }

  /** Base class for StandardScaler's [[TransformOperation]]. This class has to be extended for
    * all types which are supported by [[StandardScaler]]'s transform operation.
    *
    * @tparam T
    */
  abstract class StandardScalerTransformOperation[T: TypeInformation: ClassTag]
    extends TransformOperation[
        StandardScaler,
        (linalg.Vector[Double], linalg.Vector[Double]),
        T,
        T] {

    var mean: Double = _
    var std: Double = _

    override def getModel(
      instance: StandardScaler,
      transformParameters: ParameterMap)
    : DataSet[(linalg.Vector[Double], linalg.Vector[Double])] = {
      mean = transformParameters(Mean)
      std = transformParameters(Std)

      instance.metricsOption match {
        case Some(metrics) => metrics
        case None =>
          throw new RuntimeException("The StandardScaler has not been fitted to the data. " +
            "This is necessary to estimate the mean and standard deviation of the data.")
      }
    }

    def scale[V <: Vector: BreezeVectorConverter](
      vector: V,
      model: (linalg.Vector[Double], linalg.Vector[Double]))
    : V = {
      val (broadcastMean, broadcastStd) = model
      var myVector = vector.asBreeze
      myVector -= broadcastMean
      myVector :/= broadcastStd
      myVector = (myVector :* std) + mean
      myVector.fromBreeze
    }
  }

  /** [[TransformOperation]] to transform [[Vector]] types
    *
    * @tparam T
    * @return
    */
  implicit def transformVectors[T <: Vector: BreezeVectorConverter: TypeInformation: ClassTag] = {
    new StandardScalerTransformOperation[T]() {
      override def transform(
          vector: T,
          model: (linalg.Vector[Double], linalg.Vector[Double]))
        : T = {
        scale(vector, model)
      }
    }
  }

  /** [[TransformOperation]] to transform tuples of type ([[Vector]], [[Double]]).
    *
    * @tparam T
    * @return
    */
  implicit def transformTupleVectorDouble[
      T <: Vector: BreezeVectorConverter: TypeInformation: ClassTag] = {
    new StandardScalerTransformOperation[(T, Double)] {
      override def transform(
          element: (T, Double),
          model: (linalg.Vector[Double], linalg.Vector[Double]))
        : (T, Double) = {
        (scale(element._1, model), element._2)
      }
    }
  }

  /** [[TransformOperation]] to transform [[LabeledVector]].
    *
    */
  implicit val transformLabeledVector = new StandardScalerTransformOperation[LabeledVector] {
    override def transform(
        element: LabeledVector,
        model: (linalg.Vector[Double], linalg.Vector[Double]))
      : LabeledVector = {
      val LabeledVector(label, vector) = element

      LabeledVector(label, scale(vector, model))
    }
  }
}
