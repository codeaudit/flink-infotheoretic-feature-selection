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

import breeze.linalg
import breeze.numerics.sqrt._
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.{LabeledVector, Parameter, ParameterMap}
import org.apache.flink.ml.math.Breeze._
import org.apache.flink.ml.math.{BreezeVectorConverter, Vector, DenseVector}
import org.apache.flink.ml.pipeline.{TransformOperation, FitOperation,
Transformer}
import org.apache.flink.ml.preprocessing.FrequencyDiscretizer.Splits
import org.apache.flink.api.scala.utils.DataSetUtils
import org.apache.flink.util.XORShiftRandom

import scala.reflect.ClassTag
import scala.collection.mutable

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
class FrequencyDiscretizer extends Transformer[FrequencyDiscretizer] {

  private[preprocessing] var splits: Option[Array[Array[Float]]] = None

  /** Sets the target mean of the transformed data
    *
    * @param mu the user-specified mean value.
    * @return the StandardScaler instance with its mean value set to the user-specified value
    */
  def setSplits(splits: Array[Array[Float]]): FrequencyDiscretizer = {
    require(FrequencyDiscretizer.checkAllSplits(splits))
    parameters.add(Splits, splits)
    this
  }
}

object FrequencyDiscretizer {

  // ====================================== Parameters =============================================

  case object Splits extends Parameter[Array[Array[Float]]] {
    override val defaultValue: Option[Array[Array[Float]]] = Some(Array(Array(0.0f)))
  }

  case object NBuckets extends Parameter[Int] {
    override val defaultValue: Option[Int] = Some(2)
  }

  case object Seed extends Parameter[Long] {
    override val defaultValue: Option[Long] = Some(481366818L)
  }

  // ==================================== Factory methods ==========================================

  def apply(): FrequencyDiscretizer = {
    new FrequencyDiscretizer()
  }

  // ====================================== Operations =============================================

  /** Trains the [[org.apache.flink.ml.preprocessing.StandardScaler]] by learning the mean and
    * standard deviation of the training data. These values are used inthe transform step
    * to transform the given input data.
    *
    * @tparam T Input data type which is a subtype of [[Vector]]
    * @return
    */
  implicit def fitVectorStandardScaler = new FitOperation[FrequencyDiscretizer, Vector] {
    override def fit(instance: FrequencyDiscretizer, fitParameters: ParameterMap, input: DataSet[Vector]): Unit = {
       val map = instance.parameters ++ fitParameters

      // retrieve parameters of the algorithm
      val nbuckets = map(NBuckets)
      val seed = map(Seed)
      val splits = extractSplits(input, nbuckets, seed)
      require(checkAllSplits(splits))      
      instance.splits = Some(splits)
    }
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
  private def extractSplits(dataSet: DataSet[Vector], nBuckets: Int, seed: Long) : Array[Array[Float]] = {
    val samples = getSampledInput(dataSet, nBuckets, seed)
    val candidates = findSplitCandidates(samples, nBuckets - 1)
    getSplits(candidates)
  }
  
  /**
   * Minimum number of samples required for finding splits, regardless of number of bins.  If
   * the dataset has fewer rows than this value, the entire dataset will be used.
   */
  private[flink] val minSamplesRequired: Int = 10000

  /**
   * Sampling from the given dataset to collect quantile statistics.
   */
  private[preprocessing] def getSampledInput(dataset: DataSet[Vector], numBins: Int, seed: Long): Seq[Vector] = {
    val totalSamples = dataset.count()
    require(totalSamples > 0,
      "QuantileDiscretizer requires non-empty input dataset but was given an empty input.")
    val requiredSamples = math.max(numBins * numBins, minSamplesRequired)
    val fraction = math.min(requiredSamples.toDouble / totalSamples, 1.0)
    dataset.sample(false, fraction, new XORShiftRandom(seed).nextInt()).collect()
  }
  
  
  /**
   * Compute split points with respect to the sample distribution.
   */
  private[preprocessing] def findSplitCandidates(
      samples: Seq[Vector], 
      numSplits: Int): Array[Array[Float]] = { 
    
    val result = (0 until samples(0).size).map { feat => 
          val values = samples.map(v => v(feat))        
          val valueCountMap = values.foldLeft(Map.empty[Float, Int]) { (m, x) =>
            m + ((x.toFloat, m.getOrElse(x.toFloat, 0) + 1))
          }
          val valueCounts = valueCountMap.toSeq.sortBy(_._1).toArray ++ Array((Float.MaxValue, 1))
          val possibleSplits = valueCounts.length - 1
          if (possibleSplits <= numSplits) {
            valueCounts.dropRight(1).map(_._1)
          } else {
            val stride: Double = math.ceil(samples.length.toDouble / (numSplits + 1))
            val splitsBuilder = mutable.ArrayBuilder.make[Float]
            var index = 1
            // currentCount: sum of counts of values that have been visited
            var currentCount = valueCounts(0)._2
            // targetCount: target value for `currentCount`. If `currentCount` is closest value to
            // `targetCount`, then current value is a split threshold. After finding a split threshold,
            // `targetCount` is added by stride.
            var targetCount = stride
            while (index < valueCounts.length) {
              val previousCount = currentCount
              currentCount += valueCounts(index)._2
              val previousGap = math.abs(previousCount - targetCount)
              val currentGap = math.abs(currentCount - targetCount)
              // If adding count of current value to currentCount makes the gap between currentCount and
              // targetCount smaller, previous value is a split threshold.
              if (previousGap < currentGap) {
                splitsBuilder += valueCounts(index - 1)._1
                targetCount += stride
              }
              index += 1
            }
            splitsBuilder.result()
          }
      }    
    result.toArray
  }
  
    /**
   * Adjust split candidates to proper splits by: adding positive/negative infinity to both sides as
   * needed, and adding a default split value of 0 if no good candidates are found.
   */
  private[preprocessing] def getSplits(candidates: Array[Array[Float]]): Array[Array[Float]] = {
      candidates.map{ cand => 
        val effectiveValues = if (cand.nonEmpty) {
        if (cand.head == Float.NegativeInfinity
          && cand.last == Float.PositiveInfinity) {
          cand.drop(1).dropRight(1)
        } else if (cand.head == Float.NegativeInfinity) {
          cand.drop(1)
        } else if (cand.last == Float.PositiveInfinity) {
          cand.dropRight(1)
        } else {
          cand
        }
      } else {
        cand
      }
  
      if (effectiveValues.isEmpty) {
        Array(Float.NegativeInfinity, 0, Float.PositiveInfinity)
      } else {
        Array(Float.NegativeInfinity) ++ effectiveValues ++ Array(Float.PositiveInfinity)
      }
    }
  }
  
  /** We require splits to be of length >= 3 and to be in strictly increasing order. */
  def checkAllSplits(splits: Array[Array[Float]]): Boolean = {
    splits.map{ s =>
      if (s.length < 3) {
        false
      } else {
        var i = 0
        val n = s.length - 1
        while (i < n) {
          if (s(i) >= s(i + 1)) return false
          i += 1
        }
        true
      }
    }.filter(_ == false).length > 0
  }

  /**
   * Binary searching in several buckets to place each data point.
   * @throws SparkException if a feature is < splits.head or > splits.last
   */
  private[preprocessing] def binarySearchForBuckets(splits: Array[Float], feature: Double): Double = {
    if (feature == splits.last) {
      splits.length - 2
    } else {
      val idx = java.util.Arrays.binarySearch(splits, feature.toFloat)
      if (idx >= 0) {
        idx
      } else {
        val insertPos = -idx - 1
        if (insertPos == 0 || insertPos == splits.length) {
          throw new Exception(s"Feature value $feature out of Bucketizer bounds" +
            s" [${splits.head}, ${splits.last}].  Check your features, or loosen " +
            s"the lower/upper bound constraints.")
        } else {
          insertPos - 1
        }
      }
    }
  }

  /** Base class for StandardScaler's [[TransformOperation]]. This class has to be extended for
    * all types which are supported by [[StandardScaler]]'s transform operation.
    *
    * @tparam T
    */
  abstract class FrequencyDiscretizerTransformOperation
    extends TransformOperation[
        FrequencyDiscretizer, 
        Array[Array[Float]], 
        Vector, 
        Vector] {

    var splits: Array[Array[Float]] = _

    override def getModel(
      instance: FrequencyDiscretizer, 
      transformParameters: ParameterMap): DataSet[Array[Array[Float]]] = {

      val env = ExecutionEnvironment.getExecutionEnvironment
      splits = transformParameters(Splits)

      val result = instance.splits match {
        case Some(s) => env.fromElements((0, s))
        case None =>
          throw new RuntimeException("The discretizer has not been fitted to the data. ")
      }
      result.map(_._2)
    }

    def discretize(
      vector: Vector,
      model: Array[Array[Float]])
    : Vector = {
      val splits = model
      new DenseVector((0 until vector.size).map(ifeat => binarySearchForBuckets(splits(ifeat), vector(ifeat))).toArray)
    }
  }

  /** [[TransformOperation]] to transform [[Vector]] types
    *
    * @tparam T
    * @return
    */
  implicit def transformVectors = {
    new FrequencyDiscretizerTransformOperation() {
      override def transform(
          vector: Vector,
          model: Array[Array[Float]])
        : Vector = {
        discretize(vector, model)
      }
    }
  }
}
