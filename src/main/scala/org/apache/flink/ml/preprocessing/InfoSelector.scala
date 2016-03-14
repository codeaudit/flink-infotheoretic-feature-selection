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
import breeze.numerics.sqrt
import breeze.numerics.sqrt._
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.{LabeledVector, Parameter, ParameterMap}
import org.apache.flink.ml.math.Breeze._
import org.apache.flink.ml.math.{BreezeVectorConverter, Vector}
import org.apache.flink.ml.pipeline.{TransformOperation, FitOperation,Transformer}
import org.apache.flink.ml.preprocessing.InfoSelector.{Selected}
import scala.reflect.ClassTag
import org.apache.flink.ml.math.SparseVector
import scala.collection.mutable.ArrayBuilder
import org.apache.flink.ml.math.DenseVector

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
class InfoSelector extends Transformer[InfoSelector] {

  private[preprocessing] var selectedFeatures: Option[DataSet[Int]] = None

  /** Sets the target mean of the transformed data
    *
    * @param mu the user-specified mean value.
    * @return the StandardScaler instance with its mean value set to the user-specified value
    */
  def setCriterion(s: Array[Int]): InfoSelector = {
    parameters.add(Selected, s)
    this
  }


}

object InfoSelector {

  // ====================================== Parameters =============================================

  case object Selected extends Parameter[Array[Int]] {
    override val defaultValue: Option[Array[Int]] = Some(Array(0))
  }
  

  // ==================================== Factory methods ==========================================

  def apply(): InfoSelector = {
    new InfoSelector()
  }

  // ====================================== Operations =============================================


  /** Trains the [[StandardScaler]] by learning the mean and standard deviation of the training
    * data which is of type [[LabeledVector]]. The mean and standard deviation are used to
    * transform the given input data.
    *
    */
  implicit val fitLabeledVectorInfoSelector = {
    new FitOperation[InfoSelector, LabeledVector] {
      override def fit(
          instance: InfoSelector, 
          fitParameters: ParameterMap, 
          input: DataSet[LabeledVector]): Unit = {
        
        
        val selected = extractRelevantFeatures(input)

        instance.selectedFeatures = Some(selected)
      }
    }
  }
  
    

  protected def isSorted(array: Array[Int]): Boolean = {
    var i = 1
    val len = array.length
    while (i < len) {
      if (array(i) < array(i-1)) return false
      i += 1
    }
    true
  }

  /** Base class for StandardScaler's [[TransformOperation]]. This class has to be extended for
    * all types which are supported by [[StandardScaler]]'s transform operation.
    *
    * @tparam T
    */
 abstract class InfoSelectorTransformOperation[T: TypeInformation: ClassTag]
    extends TransformOperation[
        InfoSelector,
        Array[Int],
        T,
        T] {

    var selected: Array[Int] = _

   def getModel(
      instance: InfoSelector, 
      transformParameters: ParameterMap): DataSet[Int] = {
      
      selected = transformParameters(Selected)

      instance.selectedFeatures match {
        case Some(s) => s
        case None =>
          throw new RuntimeException("The InfoSelector has not been fitted to the data. ")
      }
    }

    def compress(
        vector: Vector, 
        filterIndices: Array[Int]) 
      : Vector = {
       vector match {
        case SparseVector(size, indices, values) =>
          val newSize = filterIndices.length
          val newValues = new ArrayBuilder.ofDouble
          val newIndices = new ArrayBuilder.ofInt
          var i = 0
          var j = 0
          var indicesIdx = 0
          var filterIndicesIdx = 0
          while (i < indices.length && j < filterIndices.length) {
            indicesIdx = indices(i)
            filterIndicesIdx = filterIndices(j)
            if (indicesIdx == filterIndicesIdx) {
              newIndices += j
              newValues += values(i)
              j += 1
              i += 1
            } else {
              if (indicesIdx > filterIndicesIdx) {
                j += 1
              } else {
                i += 1
              }
            }
          }
          // TODO: Sparse representation might be ineffective if (newSize ~= newValues.size)
          new SparseVector(newSize, newIndices.result(), newValues.result())
        case DenseVector(values) =>
          //val values = features.toArray
          new DenseVector(filterIndices.map(i => vector(i)))
        case other =>
          throw new UnsupportedOperationException(
            s"Only sparse and dense vectors are supported but got ${other.getClass}.")
      }
    }
  }

  /** [[TransformOperation]] to transform [[Vector]] types
    *
    * @tparam T
    * @return
    */
  implicit def transformVectors = {
    new InfoSelectorTransformOperation[Vector]() {
      override def transform(
          vector: Vector,
          model: Array[Int])
        : Vector = {
        compress(vector, model)
      }
    }
  }

  /** [[TransformOperation]] to transform tuples of type ([[Vector]], [[Double]]).
    *
    * @tparam T
    * @return
    */
  implicit def transformTupleVectorDouble = {
    new InfoSelectorTransformOperation[(Vector, Double)] {
      override def transform(
          element: (Vector, Double),
          model: Array[Int])
        : (Vector, Double) = {
        (compress(element._1, model), element._2)
      }
    }
  }

  /** [[TransformOperation]] to transform [[LabeledVector]].
    *
    */
  implicit val transformLabeledVector = new InfoSelectorTransformOperation[LabeledVector] {
    override def transform(
        element: LabeledVector,
        model: Array[Int])
      : LabeledVector = {
      val LabeledVector(label, vector) = element

      LabeledVector(label, compress(vector, model))
    }
  }
}
