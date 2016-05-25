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

import scala.collection.mutable.ArrayBuilder
import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import breeze.numerics.sqrt._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM}
import org.apache.flink.api.common.operators.Order
import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.ml.common.{LabeledVector, Parameter, ParameterMap}
import org.apache.flink.api.scala._
import org.apache.flink.ml.math._
import org.apache.flink.ml.pipeline.{TransformOperation, FitOperation,Transformer}
import org.apache.flink.ml.preprocessing.InfoSelector.{Selected, NFeatures, Criterion}
import org.apache.flink.util.Collector
import org.apache.flink.api.scala.utils.`package`.DataSetUtils
import scala.collection.mutable.ArrayBuffer
import org.apache.flink.ml.common.FlinkMLTools

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
  
  // Case class for criteria/feature
  case class F(feat: Int, crit: Double) 
  // Case class for columnar data (dense and sparse version)
  case class ColumnarData(dense: DataSet[((Int, Int), Array[Byte])], 
      sparse: DataSet[(Int, BDV[Byte])],
      isDense: Boolean,
      originalNPart: Int)

class InfoSelector extends Transformer[InfoSelector] {

  var selectedFeatures: Option[Array[Int]] = None
  private[preprocessing] var nfeatures: Int = 10
  private[preprocessing] var criterion: String = "mrmr" 

  /** Sets the target mean of the transformed data
    *
    * @param mu the user-specified mean value.
    * @return the StandardScaler instance with its mean value set to the user-specified value
    */
  def setSelected(s: Array[Int]): InfoSelector = {
    require(InfoSelector.isSorted(s))
    parameters.add(Selected, s)
    this
  }
  
  def setNFeatures(s: Int): InfoSelector = {
    require(s > 0)
    parameters.add(NFeatures, s)
    this
  }

  def setCriterion(s: String): InfoSelector = {
    parameters.add(Criterion, s)
    this
  }
}

object InfoSelector {

  // ====================================== Parameters =============================================

  case object Selected extends Parameter[Array[Int]] {
    override val defaultValue: Option[Array[Int]] = Some(Array(0))
  }
  
  case object NFeatures extends Parameter[Int] {
    val defaultValue = Some(10)
  }
  case object Criterion extends Parameter[String] {
    val defaultValue = Some("mrmr")
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
        
        val map = instance.parameters ++ fitParameters

        // retrieve parameters of the algorithm
        val nselect = map(NFeatures)
        val critFactory = new InfoThCriterionFactory(map(Criterion))
        
        val selected = trainOn(input, critFactory, nselect)
        require(isSorted(selected))
        instance.selectedFeatures = Some(selected)
      }
    }
  }
  
  private[InfoSelector] def isSorted(array: Array[Int]): Boolean = {
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
 abstract class InfoSelectorTransformOperation extends TransformOperation[
        InfoSelector,
        Array[Int],
        LabeledVector,
        LabeledVector] {

    var selected: Array[Int] = _

   override def getModel(
      instance: InfoSelector, 
      transformParameters: ParameterMap): DataSet[Array[Int]] = {
      
      val env = ExecutionEnvironment.getExecutionEnvironment
      selected = transformParameters(Selected)
      
      val result = instance.selectedFeatures match {
        case Some(s) => env.fromElements((0, s))
        case None =>
          throw new RuntimeException("The InfoSelector has not been fitted to the data. ")
      }
      result.map(_._2)
    }

    def select(
        lp: LabeledVector, 
        filterIndices: Array[Int]) 
      : LabeledVector = {
       lp.vector match {
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
          new LabeledVector(lp.label, new SparseVector(newSize, newIndices.result(), newValues.result()))
        case DenseVector(values) =>
          new LabeledVector(lp.label, new DenseVector(filterIndices.map(i => values(i))))
      }
    }
  }

  /** [[TransformOperation]] to transform [[Vector]] types
    *
    * @tparam T
    * @return
    */
  implicit val transformLabeledVector = {
    new InfoSelectorTransformOperation() {
      override def transform(
          vector: LabeledVector,
          model: Array[Int])
        : LabeledVector = {        
        select(vector, model)
      }
    }
  }
  
  /*** THE START OF THE ALGORITHM LOGIC ***/


  /**
   * Performs a info-theory FS process.
   * 
   * @param data Columnar data (last element is the class attribute).
   * @param nInstances Number of samples.
   * @param nFeatures Number of features.
   * @return A list with the most relevant features and its scores.
   * 
   */
  private[preprocessing] def selectDenseFeatures(
      data: ColumnarData,
      criterionFactory: InfoThCriterionFactory, 
      nToSelect: Int,
      nInstances: Long,
      nFeatures: Int) = {
    
    val label = nFeatures - 1
    // Initialize all criteria with the relevance computed in this phase. 
    // It also computes and saved some information to be re-used.
    val it = InfoTheory.initializeDense(label, nInstances, nFeatures, data.originalNPart)
    val (mt, jt, rev, fcol, counter)  = it.initialize(data.dense)

    // Initialize all (except the class) criteria with the relevance values
    val pool = Array.fill[InfoThCriterion](nFeatures - 1) {
      val crit = criterionFactory.getCriterion.init(Float.NegativeInfinity)
      crit.setValid(false)
    }    
    rev.collect().foreach{ case (x, mi) => 
      pool(x) = criterionFactory.getCriterion.init(mi.toFloat) 
    }
    
    // Print most relevant features
    val topByRelevance = rev.groupBy(1).sortGroup(1, Order.DESCENDING).first(nToSelect).collect()
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

      val redundancies = it.getRedundancies(data.dense, selected.head.feat, mt, jt, fcol, counter)      
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
   * Performs a info-theory FS process.
   * 
   * @param data Columnar data (last element is the class attribute).
   * @param nInstances Number of samples.
   * @param nFeatures Number of features.
   * @return A list with the most relevant features and its scores.
   * 
   */
  private[preprocessing] def selectSparseFeatures(
      data: ColumnarData,
      criterionFactory: InfoThCriterionFactory, 
      nToSelect: Int,
      nInstances: Long,
      nFeatures: Int) = {
    
    val label = nFeatures - 1
    // Initialize all criteria with the relevance computed in this phase. 
    // It also computes and saved some information to be re-used.
    val it = InfoTheory.initializeSparse(label, nInstances, nFeatures)
    val (mt, jt, rev, fcol)  = it.initialize(data.sparse)

    // Initialize all (except the class) criteria with the relevance values
    val pool = Array.fill[InfoThCriterion](nFeatures - 1) {
      val crit = criterionFactory.getCriterion.init(Float.NegativeInfinity)
      crit.setValid(false)
    }    
    rev.collect().foreach{ case (x, mi) => 
      pool(x) = criterionFactory.getCriterion.init(mi.toFloat) 
    }
    
    // Print most relevant features
    val topByRelevance = rev.groupBy(1).sortGroup(1, Order.DESCENDING).first(nToSelect).collect()
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

      val red = it.getRedundancies(data.sparse, mt, jt, fcol, selected.head.feat)      
      // Update criteria with the new redundancy values
      red.collect().par.foreach({case (k, (mi, cmi)) =>
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
   * @param data DataSet of LabeledPoint.
   * @return A feature selection model which contains a subset of selected features.
   * 
   */
  def trainOn(data: DataSet[LabeledVector], 
      criterionFactory: InfoThCriterionFactory, 
      nToSelect: Int): Array[Int] = {
      
    // Feature vector must be composed of bytes, not the class
    println("Initializing feature selector...")
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
        throw new IllegalArgumentException(s"Info-Theoretic Framework requires positive values in range [0, 255]")
      }           
    }
    
    // Get basic info
    val first = data.first(1).collect()(0)
    val dense = first.vector.isInstanceOf[DenseVector]    
    val nInstances = data.count()
    val nFeatures = first.vector.size + 1
    val oldNP = data.mapPartition(it => Seq(1)).reduce{_ + _}.collect()(0)
    val classMap = data.map(_.label).distinct.collect()
        .zipWithIndex.map(t => t._1 -> t._2.toByte)
        .toMap
    
    require(nToSelect < nFeatures)  
    
    // Start the transformation to the columnar format
    val colData = if(dense) {
      println("Performing the columnar transformation...")
      // Transform data into a columnar format by transposing the local matrix in each partition
      val denseIndexing = new RichMapPartitionFunction[LabeledVector, ((Int, Int), Array[Byte])]() {        
          def mapPartition(it: java.lang.Iterable[LabeledVector], out: Collector[((Int, Int), Array[Byte])]): Unit = {
            val index = getRuntimeContext().getIndexOfThisSubtask() // Partition index
            val mat = for (i <- 0 until nFeatures) yield new scala.collection.mutable.ListBuffer[Byte]
            for(reg <- it.asScala) {
              for (i <- 0 until (nFeatures - 1)) mat(i) += reg.vector(i).toByte
              mat(nFeatures - 1) += reg.label.toByte
            }
            for(i <- 0 until nFeatures) out.collect((i, index) -> mat(i).toArray) // numPartitions
          }
        }
        val denseData = data.mapPartition(denseIndexing).partitionByRange(0)
      val colpdata = FlinkMLTools.persist(denseData, "/tmp/dense-flink-columnar")      
      ColumnarData(colpdata, null, true, oldNP)    
      
    } else {   
        
      // Transform data into a columnar format by transposing the local matrix in each partition
      val sparseIndexing = new RichMapPartitionFunction[(Long, LabeledVector), (Int, (Long, Byte))]() {        
        def mapPartition(it: java.lang.Iterable[(Long, LabeledVector)], out: Collector[(Int, (Long, Byte))]): Unit = {
          for((iinst, lp) <- it.asScala) {
            requireByteValues(lp.vector)
            val sv = lp.vector.asInstanceOf[SparseVector]
            val output = (nFeatures - 1) -> (iinst, classMap(lp.label))
            out.collect(output)
            for(i <- 0 until sv.indices.length) 
                out.collect(sv.indices(i) -> (iinst, sv.data(i).toByte))
          }
        }
      }
      val sparseData = data.zipWithIndex.mapPartition(sparseIndexing)
      
      // Transform sparse data into a columnar format 
      // by grouping all values for the same feature in a single vector
      val columnarData = sparseData.groupBy(0).reduceGroup{ it =>
          //val a = it.toArray
          var k: Int = -1
          val init = Array.fill[Byte](nInstances.toInt)(0)
          val result: BDV[Byte] = new BDV(init)
          //if(a.size >= nInstances) {
          for((fk, (iind, v)) <- it){
            k = fk
            result(iind.toInt) = v
          }
          k -> result // Feature index -> array of cells
        }.partitionByHash(0)
      val colpdata = FlinkMLTools.persist(columnarData, "/tmp/sparse-flink-columnar")
      ColumnarData(null, colpdata, false, oldNP)
    }
    
    // Start the main algorithm
    println("Starting to select features...")
    val selected = if(dense) {
      selectDenseFeatures(colData, criterionFactory, nToSelect, 
        nInstances, nFeatures)
    } else {
      selectSparseFeatures(colData, criterionFactory, nToSelect, 
        nInstances, nFeatures)
    }
  
    // Print best features according to the mRMR measure
    val out = selected.map{case F(feat, rel) => 
        (feat + 1) + "\t" + "%.4f".format(rel)
      }.mkString("\n")
    println("\n*** Selected features ***\nFeature\tScore\n" + out)
    // Features must be sorted
    selected.map{case F(feat, rel) => feat}.sorted.toArray
  }
}
