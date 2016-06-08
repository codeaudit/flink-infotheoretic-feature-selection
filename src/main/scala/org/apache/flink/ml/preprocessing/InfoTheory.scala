/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.preprocessing

import breeze.linalg._
import breeze.numerics._
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM, Matrix => BM}
import scala.collection.mutable
import scala.collection.JavaConverters._
import org.apache.flink.api.scala._
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.ml.common.{LabeledVector, Parameter, ParameterMap}
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml.math.{DenseMatrix, DenseVector}
import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.util.Collector
import scala.collection.mutable.ArrayBuffer
import org.apache.flink.ml.common.FlinkMLTools
import org.apache.flink.api.common.functions.RichFilterFunction


/*
 * Basic and distributed primitives for Info-Theory computations: Mutual Information (MI)
 * and Conditional Mutual Information (CMI). These are adapted to compute and re-used some 
 * information between processes according to the generic formula proposed by Brown et al. in [1].
 * Data must be in a columnar format. 
 *
 * [1] Brown, G., Pocock, A., Zhao, M. J., & Lujn, M. (2012). 
 * "Conditional likelihood maximization: a unifying framework 
 * for information theoretic feature selection." 
 * The Journal of Machine Learning Research, 13(1), 27-66.
 * 
 *
 */

class InfoTheory {
  
  /**
   * Computes MI between two variables using histograms as input data.
   * 
   * @param data DataSet of tuples (feature, 2-dim histogram).
   * @param yProb Vector of proportions for the secondary feature.
   * @param nInstances Number of instances.
   * @result A DataSet of tuples (feature, MI).
   * 
   */
  protected def computeMutualInfo(
      data: DataSet[(Int, Array[Array[Long]])],
      yProb: DataSet[Array[Float]],
      nInstances: Long) = {   
    
    val result = data.map( new RichMapFunction[(Int, Array[Array[Long]]), (Int, Float)]() {
      var byProb: Array[Float] = null

      override def open(config: Configuration): Unit = {
        // 3. Access the broadcasted DataSet as a Collection
        byProb = getRuntimeContext().getBroadcastVariable[Array[Float]]("byProb").asScala(0)
      }
  
      def map(tuple: (Int, Array[Array[Long]])): (Int, Float) = {
        var mi = 0.0d
        val m = tuple._2
        // Aggregate by row (x)
        val xProb = m.map(_.sum).map(_.toFloat / nInstances)
        for(i <- 0 until m.length){
          for(j <- 0 until m(i).length){
            val pxy = m(i)(j).toFloat / nInstances
            val py = byProb(j); val px = xProb(i)
            // To avoid NaNs
            if(pxy != 0 && px != 0 && py != 0){
              mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
            }             
          }
        } 
        (tuple._1, mi.toFloat)
      }
      
              
    }).withBroadcastSet(yProb, "byProb")
    result
  }
  
  /**
   * Computes MI and CMI between three variables using histograms as input data.
   * 
   * @param data DataSet of tuples (feature, 3-dim histogram).
   * @param varY Index of the secondary feature.
   * @param varZ Index of the conditional feature.
   * @param marginalProb DataSet of tuples (feature, marginal vector)
   * @param jointProb DataSet of tuples (feature, joint matrices)
   * @param n Number of instances.*
   * @result A DataSet of tuples (feature, CMI).
   * 
   */
  protected def computeConditionalMutualInfo(
      data: DataSet[(Int, BDV[BDM[Long]])],
      varY: DataSet[(Int, Float)],
      varZ: Int,
      marginalProb: DataSet[(Int, Array[Float])],
      jointProb: DataSet[(Int, Array[Array[Float]])],
      n: Long) = {
    
    val yProb = marginalProb.filter( new FilterWitH2).withBroadcastSet(varY, "vary").map(c => c._1 -> Array(c._2))
    val zProb = marginalProb.filter(_._1 == varZ).map(c => c._1 -> Array(c._2))
    val yzProb = jointProb.filter( new FilterWitH4).withBroadcastSet(varY, "vary").map(c => -1 -> c._2)
    val dprob = yProb.union(zProb).union(yzProb)
    
    val result = data.map( new RichMapFunction[(Int, BDV[BDM[Long]]), (Int, (Float, Float))]() {
      var prob: Array[(Int, Array[Array[Float]])] = null

      override def open(config: Configuration): Unit = {
        // 3. Access the broadcasted DataSet as a Collection
        val aux = getRuntimeContext()
          .getBroadcastVariable[(Int, Array[Array[Float]])]("broadVar")
          .asScala
          .toArray
        prob = aux
      }
      
      def map(tuple: (Int, BDV[BDM[Long]])): (Int, (Float, Float)) = {
        
        var cmi = 0.0d; var mi = 0.0d
        val m = tuple._2
        // Aggregate values by row (X)
        val aggX = m.map(h1 => sum(h1(*, ::)).toDenseVector)
        // Use the previous variable to sum up and so obtaining X accumulators 
        val xProb = aggX.reduce(_ + _).apply(0).map(_.toFloat / n)
        // Aggregate all matrices in Z to obtain the joint probabilities for X and Y
        val xyProb = m.reduce(_ + _).apply(0).map(_.toFloat / n)  
        val xzProb = aggX.map(_.map(_.toFloat / n))
        val bzProb = prob.filter(_._1 == varZ)(0)._2(0)
        val byzProb = prob.filter(_._1 == -1)(0)._2
        val byProb = prob.filter(c => c._1 != -1 && c._1 != varZ)(0)._2(0)
        
        for(z <- 0 until m.length){
          for(x <- 0 until m(z).rows){
            for(y <- 0 until m(z).cols) {
              val pz = bzProb(z) // create a breeze dense vector for computations
              val pxyz = (m(z)(x, y).toFloat / n) / pz
              val pxz = xzProb(z)(x) / pz
              val pyz = byzProb(y)(z) / pz
              if(pxz != 0 && pyz != 0 && pxyz != 0) {
                cmi += pz * pxyz * (math.log(pxyz / (pxz * pyz)) / math.log(2))
              }              
              if (z == 0) { // Do MI computations only once
                val px = xProb(x); val pxy = xyProb(x, y); val py = byProb(y)
                if(pxy != 0 && px != 0 && py != 0) {
                  mi += pxy * (math.log(pxy / (px * py)) / math.log(2))
                }                
              }
            }            
          }
        } 
        tuple._1 -> (mi.toFloat, cmi.toFloat)
        
      }
               
    }).withBroadcastSet(dprob, "broadVar")
    
    result
  }
  
}

/**
 * Class that computes histograms for the Info-Theory primitives and starts 
 * the selection process (sparse and high-dimensional version).
 *
 * The constructor method caches a single attribute (usually the class) to be 
 * re-used in the next iterations as conditional variable. This also computes 
 * and caches the relevance values, and the marginal and joint proportions derived 
 * from this operation.
 * 
 * @param data DataSet of tuples (feature, values).
 * @param fixedFeat Index of the fixed attribute (usually the class). 
 * @param nInstances Number of samples.
 * @param nFeatures Number of features.
 *
 */
class InfoTheorySparse ( 
    fixedFeat: Int,
    nInstances: Long,      
    nFeatures: Int) extends InfoTheory with Serializable {
  

  
  def initialize(data: DataSet[(Int, BDV[Byte])]) = {
    // Broadcast the class attribute (fixed)
    val fixedVal = data.filter(_._1 == fixedFeat).collect()(0)._2
    val fixedCol = (fixedFeat, fixedVal)
    val fixedColHistogram = computeFrequency(fixedVal, nInstances)
    
      // Compute and cache the relevance values, and the marginal and joint proportions
    val triple = {
      val histograms = computeHistograms(data, fixedCol._2, fixedColHistogram)
      val jointTable = histograms.map(h => h._1 -> h._2.map(_.map(_.toFloat / nInstances)))
      val marginalTable = jointTable.map(h => h._1 -> h._2.map(_.sum))
      
      // Remove the class attribute from the computations
      val label = nFeatures - 1 
      val fdata = histograms.filter(_._1 != label)
      val marginalY = marginalTable.filter(_._1 == fixedFeat).map(_._2)
      
      // Compute MI between all input features and the class (relevances)
      val relevances = computeMutualInfo(fdata, marginalY, nInstances)
      val (mt, jt, rev) = FlinkMLTools.persist(marginalTable, jointTable, relevances, 
          "hdfs://bigdata:8020/tmp/marginal-prob", "hdfs://bigdata:8020/tmp/joint-prob", "hdfs://bigdata:8020/tmp/relev-prob")
      (mt, jt, rev, fixedVal)
      
      
    } 
    triple
  }
  
  private def computeFrequency(data: BDV[Byte], nInstances: Long) = {
    val tmp = data.activeValuesIterator.toArray
      .groupBy(l => l).map(t => (t._1, t._2.size.toLong))
    val lastElem = (0: Byte, nInstances - tmp.filter({case (v, _) => v != 0}).values.sum)
    tmp + lastElem
  }
  
    /**
   * Computes simple and conditional redundancy for all input attributes with respect to 
   * a secondary variable (Y) and a conditional variable (already cached).
   * 
   * @param varY Index of the secondary feature (class).
   * @result A DataSet of tuples (feature, (redundancy, conditional redundancy)).
   * 
   */
  def getRedundancies(data: DataSet[(Int, BDV[Byte])],
      marginalProb: DataSet[(Int, Array[Float])], 
      jointProb: DataSet[(Int, Array[Array[Float]])],
      fixedCol: BDV[Byte],
      varY: DataSet[(Int, Float)]) = {
    
    // Get and broadcast Y and the fixed variable
    val ycol = data.filter(_._1 == varY).collect()(0)._2
    val varZ = fixedFeat
    
    // Compute conditional histograms for all variables with respect to Y and the fixed variable
    val histograms3d = computeConditionalHistograms(data, ycol, (fixedFeat, fixedCol))
        .filter(h => h._1 != varZ && h._1 != varY)
    // Compute CMI and MI for all input variables with respect to Y and Z
    computeConditionalMutualInfo(histograms3d, varY, varZ, 
        marginalProb, jointProb, nInstances)
 }
  
    /**
   * Computes 2-dim histograms for all input attributes with respect to 
   * a secondary variable (class).
   * 
   * @param DataSet of tuples (feature, values)
   * @param ycol (feature, values).
   * @param yhist Histogram for variable Y (class).
   * 
   * @result A DataSet of tuples (feature, histogram).
   * 
   */
  private def computeHistograms(
      filterData:  DataSet[(Int, BDV[Byte])],
      ycol: BDV[Byte],
      yhist: Map[Byte, Long]) = {
    
    // Distinct values for Y
    val ys = if(ycol.size > 0) ycol.activeValuesIterator.max + 1 else 1
    val env = ExecutionEnvironment.getExecutionEnvironment
    val ydcol = env.fromCollection(ycol.toArray)
    
    val result = filterData.map( new RichMapFunction[(Int, BDV[Byte]), (Int, Array[Array[Long]])]() {
      var bycol: Array[Byte] = null

      override def open(config: Configuration): Unit = {
        // 3. Access the broadcasted DataSet as a Collection
        bycol = getRuntimeContext().getBroadcastVariable[Byte]("bycol").asScala.toArray
      }
      
      def map(tuple: (Int, BDV[Byte])): (Int, Array[Array[Long]]) = {
        val xs = if(tuple._2.size > 0) tuple._2.activeValuesIterator.max + 1 else 1 
        val result = Array.ofDim[Long](xs, ys)
        
        val histCls = mutable.HashMap.empty ++= yhist // clone
        for ((inst, x) <- tuple._2.activeIterator){
          val y = bycol(inst)  
          histCls += y -> (histCls(y) - 1)
          result(tuple._2(inst))(y) += 1
        }
        // Zeros count
        histCls.foreach({ case (c, q) => result(0)(c) += q })
        tuple._1 -> result
      }
              
    }).withBroadcastSet(ydcol, "bycol");
    
    result
  }
  
  
  /**
   * Computes 3-dim histograms for all input attributes with respect to 
   * a secondary variable and the conditional variable. Conditional feature 
   * (class) must be already cached. 
   * 
   * @param filterData DataSet of tuples (feature, values)
   * @param ycol (feature, value vector).
   * 
   * @result A DataSet of tuples (feature, histogram).
   * 
   */
  private def computeConditionalHistograms(
    filterData: DataSet[(Int, BDV[Byte])],
    ycol: BDV[Byte],
    zcol: (Int, BDV[Byte])) = {         
      
      val env = ExecutionEnvironment.getExecutionEnvironment
      
      // Compute the histogram for variable Y and get its values. 
      val yhist = new mutable.HashMap[(Byte, Byte), Long]()
      ycol.activeIterator.foreach({case (inst, y) => 
          val z = zcol._2(inst)
          yhist += (y, z) -> (yhist.getOrElse((y, z), 0L) + 1)
      })      
            
      // Get the vector for the conditional variable and compute its histogram
      val zhist = computeFrequency(zcol._2, nInstances)
      
      // Get the maximum sizes for both single variables
      val ys = if(ycol.size > 0) ycol.activeValuesIterator.max + 1 else 1
      val zs = if(zcol._2.size > 0) zcol._2.activeValuesIterator.max + 1 else 1
      
      val ydcol = env.fromElements((ycol.toArray, zcol._2.toArray, yhist.toMap, zhist))
      
      // Map operation to compute histogram per feature
      val result = filterData.map( new RichMapFunction[(Int, BDV[Byte]), (Int, BDV[BDM[Long]])]() {
        var bycol: Array[Byte] = null
        var bzcol: Array[Byte] = null
        var byhist: Map[(Byte, Byte), Long] = null
        var bzhist: Map[Byte, Long] = null
        
        override def open(config: Configuration): Unit = {
          // 3. Access the broadcasted DataSet as a Collection
          val aux = getRuntimeContext()
            .getBroadcastVariable[(Array[Byte], Array[Byte], Map[(Byte, Byte), Long], Map[Byte, Long])]("broadVar")
            .asScala.toArray
          bycol = aux(0)._1; bzcol = aux(0)._2; byhist = aux(0)._3; bzhist = aux(0)._4;
        }
        
        def map(tuple: (Int, BDV[Byte])): (Int, BDV[BDM[Long]]) = {
           // Initialization
          val xs = if(tuple._2.size > 0) tuple._2.activeValuesIterator.max + 1 else 1
          val result = BDV.fill[BDM[Long]](zs){
            BDM.zeros[Long](xs, ys)
          }
          
          // Computations for all elements in X also appearing in Y        
          val yzhist = mutable.HashMap.empty ++= byhist
          for ((inst, x) <- tuple._2.activeIterator){     
            val y = bycol(inst); val z = bzcol(inst)
            if(y != 0) yzhist += (y, z) -> (yzhist((y,z)) - 1)
            result(z)(tuple._2(inst), y) += 1
          }
          
          // Computations for non-zero elements in Y and not appearing in X
          yzhist.foreach({case ((y, z), q) => result(z)(0, y) += q})
          
          // Computations for Z elements with X and Y equal to zero
          bzhist.map({ case (zval, _) => 
            val rest = bzhist(zval) - sum(result(zval))
            result(zval)(0, 0) += rest
          })
          tuple._1 -> result
        }
    }).withBroadcastSet(ydcol, "broadVar");
    
    result
  }
}


/**
 * Class that computes histograms for the Info-Theory primitives and starts 
 * the selection process (dense version).
 *
 * The constructor method caches a single attribute (usually the class) to be 
 * re-used in the next iterations as conditional variable. This also computes 
 * and caches the relevance values, and the marginal and joint proportions derived 
 * from this operation.
 * 
 * @param data DataSet of tuples (feature, values).
 * @param fixedFeat Index of the fixed attribute (usually the class). 
 * @param nInstances Number of samples.
 * @param nFeatures Number of features.
 *
 */
class InfoTheoryDense (fixedFeat: Int,
    nInstances: Long,      
    nFeatures: Int) extends InfoTheory with Serializable {  
    
  def initialize(data: DataSet[((Int, Int), Array[Byte])]) = {
    // Count the number of distinct values per feature to limit the size of matrices
    val counterByFeat = data.map(t => if(!t._2.isEmpty) (t._1._1, t._2.max + 1) else (t._1._1, 1))
            .groupBy(_._1)
            .reduce{ (v1, v2) => if(v1._2 > v2._2) v1 else v2 }
            .collect()
            .toMap
            
    // Compute and cache the relevance values, and the marginal and joint proportions derived
    val histograms = computeHistograms(data, fixedFeat, counterByFeat)
    val ni = nInstances
    val jointTable = histograms.map(h => h._1 -> h._2.map(_.map(_.toFloat / ni)))
    val marginalTable = jointTable.map(h => h._1 -> h._2.map(_.sum))    
    
    // Remove output feature from the computations and compute MI with respect to the fixed var
    val key = fixedFeat
    val fdata = histograms.filter{t => t._1 != key}
    val marginalY = marginalTable.filter{t => t._1 == key}.map(_._2)
    val relevances = computeMutualInfo(fdata, marginalY, nInstances)
    val (mt, jt, rev) = FlinkMLTools.persist(marginalTable, jointTable, relevances, "hdfs://bigdata:8020/tmp/marginal-prob", "hdfs://bigdata:8020/tmp/joint-prob", "hdfs://bigdata:8020/tmp/relev-prob")

    (mt, jt, rev, counterByFeat)    
  }
  
  /**
   * Computes simple and conditional redundancy for all input attributes with respect to 
   * a secondary variable (Y) and a conditional variable (already cached).
   * 
   * @param varY Index of the secondary feature (class).
   * @result A DataSet of tuples (feature, (redundancy, conditional redundancy)).
   * 
   */
  def getRedundancies(data: DataSet[((Int, Int), Array[Byte])],
      varY: DataSet[(Int, Float)], 
      marginalProb: DataSet[(Int, Array[Float])],
      jointProb: DataSet[(Int, Array[Array[Float]])],
      counterByFeat: Map[Int, Int]) = {
    
    // Get and broadcast Y and the fixed variable (conditional)
    
    // Compute histograms for all variables with respect to Y and the fixed variable
    val histograms3d = computeConditionalHistograms(data, varY, fixedFeat, counterByFeat)
        .filter(new FilterWitH3).withBroadcastSet(varY, "vary")
        .filter{ h => h._1 != fixedFeat}
      
    // Compute CMI and MI for all histograms with respect to two variables
    computeConditionalMutualInfo(histograms3d, varY, fixedFeat, 
        marginalProb, jointProb, nInstances)
  }  
  
  /**
   * Computes 2-dim histograms for all input attributes 
   * with respect to a secondary variable (class).
   * 
   * @param DataSet of tuples (feature, values)
   * @param ycol (feature, values).
   * @param yhist Histogram for variable Y (class).
   * 
   * @result A DataSet of tuples (feature, histogram).
   * 
   */
  private def computeHistograms(
      data:  DataSet[((Int, Int), Array[Byte])],
      yfeat: Int,
      counter: Map[Int, Int]) = {
    
    val env = ExecutionEnvironment.getExecutionEnvironment
    val maxSize = 256
    val bycol = data.filter{ t => t._1._1 == yfeat }.map(c => c._1._2 -> c._2)
    val ys = counter.getOrElse(yfeat, maxSize).toInt
    
    val func = new RichMapPartitionFunction[((Int, Int), Array[Byte]), (Int, BDM[Long])]() {
      var ycol: Array[(Int, Array[Byte])] = null

      override def open(config: Configuration): Unit = {
        // 3. Access the broadcasted DataSet as a Collection
        ycol = getRuntimeContext().getBroadcastVariable[(Int, Array[Byte])]("bycol").asScala.toArray.sortBy(_._1)
      }
  
      def mapPartition(values: java.lang.Iterable[((Int, Int), Array[Byte])], out: Collector[(Int, BDM[Long])]): Unit = {
        var result = Map.empty[Int, BDM[Long]]
        // For each feature and block, this generates a histogram (a single matrix)
        for(((feat, block), arr) <- values.asScala) {
          val m = result.getOrElse(feat, 
              BDM.zeros[Long](counter.getOrElse(feat, maxSize).toInt, ys)) 
          for(i <- 0 until arr.length) {
            val y = ycol(block)._2(i)
            val x = arr(i)
            m(x,y) += 1
          }
          out.collect(feat -> m)
        }
      }    
    }    
    
    data.mapPartition(func).withBroadcastSet(bycol, "bycol")
      .groupBy(_._1)
      .reduce{ (v1, v2) => v1._1 -> (v1._2 + v2._2) }
      .map{ t => 
        val h = t._2
        val mat = Array.ofDim[Long](h.rows, h.cols)
          for(i <- 0 until h.rows) {
            for(j <- 0 until h.cols) {
              mat(i)(j) = h(i,j) 
            }
          }
        t._1 -> mat 
      } // Then, those histograms with the same key are aggregated
  }
  
  
  /**
   * Computes 3-dim histograms for all input attributes with respect to 
   * a secondary variable and the conditional variable. Conditional feature 
   * (class) must be already cached. 
   * 
   * @param filterData DataSet of tuples (feature, values)
   * @param ycol (feature, value vector).
   * 
   * @result A DataSet of tuples (feature, histogram).
   * 
   */
  private def computeConditionalHistograms(
    data: DataSet[((Int, Int), Array[Byte])],
    varY: DataSet[(Int, Float)],
    varZ: Int,
    counterByFeat: Map[Int, Int]) = {
    
      val ys = varY.map(c => (-1,-1) -> Array(counterByFeat.getOrElse(c._1, 256).toByte))
      val zs = counterByFeat.getOrElse(varZ, 256)
      val dzcol = data.filter{ c => c._1._1 == varZ }.map(c => (-3 -> c._1._2) -> c._2)
      val dycol = data.filter{ new FilterWitH }.withBroadcastSet(varY, "vary").map(c => (-2 -> c._1._2) -> c._2)
      val broadvar = dycol.union(ys).union(dzcol)
      
      val func = new RichMapPartitionFunction[((Int, Int), Array[Byte]), (Int, BDV[BDM[Long]])]() {
        var bycol: Array[(Int, Array[Byte])] = null
        var bzcol: Array[(Int, Array[Byte])] = null
        var ys: Int = -1
        
        override def open(config: Configuration): Unit = {
          // 3. Access the broadcasted DataSet as a Collection
          val aux = getRuntimeContext()
            .getBroadcastVariable[((Int, Int), Array[Byte])]("broadvar")
            .asScala.toArray
            
          ys = aux.filter(_._1._1 == -1)(0)._2(0).toInt
          bycol = aux.filter(_._1._1 == -2).map(c => c._1._2 -> c._2).sortBy(_._1)
          bzcol = aux.filter(_._1._1 == -3).map(c => c._1._2 -> c._2).sortBy(_._1)
          
        }
        
        def mapPartition(values: java.lang.Iterable[((Int, Int), Array[Byte])], out: Collector[(Int, BDV[BDM[Long]])]): Unit = {
          var result = Map.empty[Int, BDV[BDM[Long]]]
          // For each feature and block, it generates a 3-dim histogram (several matrices)
          for(((feat, block), arr) <- values.asScala) {
              // We create a vector (z) of matrices (x,y) to represent a 3-dim matrix
              val m = result.getOrElse(feat, 
                  BDV.fill[BDM[Long]](zs){BDM.zeros[Long](counterByFeat.getOrElse(feat, 256), ys)})
              for(i <- 0 until arr.length){
                val y = bycol(block)._2(i)
                val z = bzcol(block)._2(i)
                m(z)(arr(i), y) += 1
              }
              result += feat -> m
            }
            for(hist <- result.toIterator) out.collect(hist)
        }
      }
      // Map operation to compute histogram per feature
      val hist = data.mapPartition(func).withBroadcastSet(broadvar, "broadvar")
         
      // Matrices are aggregated
      hist.groupBy(_._1).reduce{ (h1, h2) => h1._1 -> (h1._2 + h2._2) }
  }
  
  
}

object InfoTheory {
  
  /**
   * Creates an Info-Theory object to compute MI and CMI using a greedy approach (sparse version). 
   * This apply this primitives to all the input attributes with respect to a fixed variable
   * (typically the class) and a secondary (changing) variable, typically the last selected feature.
   *
   * @param   data RDD of tuples in columnar format (feature, vector).
   * @param   fixedFeat Index of the fixed attribute (usually the class).
   * @param   nInstances Number of samples.
   * @param   nFeatures Number of features.
   * @return  An info-theory object which contains the relevances and some proportions cached.
   * 
   */
  def initializeSparse(fixedFeat: Int,
    nInstances: Long,      
    nFeatures: Int) = {
      new InfoTheorySparse(fixedFeat, nInstances, nFeatures)
  }
  
  /**
   * Creates an Info-Theory object to compute MI and CMI using a greedy approach (dense version). 
   * This apply this primitives to all the input attributes with respect to a fixed variable
   * (typically the class) and a secondary (changing) variable, typically the last selected feature.
   *
   * @param   data RDD of tuples in columnar format (feature, (block, vector)).
   * @param   fixedFeat Index of the fixed attribute (usually the class).
   * @param   nInstances Number of samples.
   * @param   nFeatures Number of features.
   * @return  An info-theory object which contains the relevances and some proportions cached.
   * 
   */
  def initializeDense(fixedFeat: Int,
    nInstances: Long,      
    nFeatures: Int) = {
      new InfoTheoryDense(fixedFeat, nInstances, nFeatures)
  }
  
  private val log2 = { x: Double => math.log(x) / math.log(2) } 
  
  /**
   * Calculate entropy for the given frequencies.
   *
   * @param freqs Frequencies of each different class
   * @param n Number of elements
   * 
   */
  private[preprocessing] def entropy(freqs: Seq[Long], n: Long) = {
    freqs.aggregate(0.0)({ case (h, q) =>
      h + (if (q == 0) 0  else (q.toDouble / n) * (math.log(q.toDouble / n) / math.log(2)))
    }, { case (h1, h2) => h1 + h2 }) * -1
  }

  /**
   * Calculate entropy for the given frequencies.
   *
   * @param freqs Frequencies of each different class
   */
  private[preprocessing] def entropy(freqs: Seq[Long]): Double = {
    entropy(freqs, freqs.reduce(_ + _))
  }
  
  
  
}

final class FilterWitH extends RichFilterFunction[((Int, Int), Array[Byte])] {
    private var y: Array[(Int, Float)] = null

    /** Reads the centroid values from a broadcast variable into a collection. */
    override def open(parameters: Configuration) {
      y = getRuntimeContext.getBroadcastVariable[(Int, Float)]("vary").asScala.toArray
    }

    def filter(p: ((Int, Int), Array[Byte])): Boolean = {
      p._1._1 == y(0)._1
    }

  }


final class FilterWitH2 extends RichFilterFunction[(Int, Array[Float])] {
    private var y: Array[(Int, Float)] = null

    /** Reads the centroid values from a broadcast variable into a collection. */
    override def open(parameters: Configuration) {
      y = getRuntimeContext.getBroadcastVariable[(Int, Float)]("vary").asScala.toArray
    }

    def filter(p: (Int, Array[Float])): Boolean = {
      p._1 == y(0)._1
    }

  }


final class FilterWitH3 extends RichFilterFunction[(Int, breeze.linalg.DenseVector[breeze.linalg.DenseMatrix[Long]])] {
    private var y: Array[(Int, Float)] = null

    /** Reads the centroid values from a broadcast variable into a collection. */
    override def open(parameters: Configuration) {
      y = getRuntimeContext.getBroadcastVariable[(Int, Float)]("vary").asScala.toArray
    }

    def filter(p: (Int, breeze.linalg.DenseVector[breeze.linalg.DenseMatrix[Long]])): Boolean = {
      p._1 != y(0)._1
    }

  }

final class FilterWitH4 extends RichFilterFunction[(Int, Array[Array[Float]])] {
    private var y: Array[(Int, Float)] = null

    /** Reads the centroid values from a broadcast variable into a collection. */
    override def open(parameters: Configuration) {
      y = getRuntimeContext.getBroadcastVariable[(Int, Float)]("vary").asScala.toArray
    }

    def filter(p: (Int, Array[Array[Float]])): Boolean = {
      p._1 == y(0)._1
    }

  }

