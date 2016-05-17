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

package test

import org.apache.flink.api.common.functions._
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import scala.collection.JavaConverters._
import org.apache.flink.ml.preprocessing.FrequencyDiscretizer
import org.apache.flink.ml.preprocessing.InfoSelector
import org.apache.flink.ml.MLUtils
import org.apache.flink.ml.common.ParameterMap
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.util.Collector
import org.apache.flink.api.java.io.PrintingOutputFormat
import org.apache.flink.ml.common.FlinkMLTools


object ErrorTest {

  def main(args: Array[String]) {

    val params: ParameterTool = ParameterTool.fromArgs(args)

    // set up execution environment
    val env = ExecutionEnvironment.getExecutionEnvironment
    
     env.getConfig.disableObjectReuse()

    // make parameters available in the web interface
    env.getConfig.setGlobalJobParameters(params)
    val training = MLUtils.readLibSVM(env, "/home/sramirez/datasets/a1a.txt")
    
    val nFeatures = training.first(1).collect()(0).vector.size + 1
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
    
    val transposed = training.mapPartition(denseIndexing)
    val tr = FlinkMLTools.persist(transposed, "/tmp/transposed")
    
    /*val a = transposed.filter(_._1._1 == (nFeatures - 1)).map(t => t._1 ->
    t._2.size).reduceGroup(_.mkString(",")).output(new
    PrintingOutputFormat())
    val b = transposed.filter(_._1._1 == 10).map(t => t._1 ->
    t._2.size).reduceGroup(_.mkString(",")).output(new
    PrintingOutputFormat())
    val c = transposed.filter(_._1._1 == 12).map(t => t._1 ->
    t._2.size).reduceGroup(_.mkString(",")).output(new
    PrintingOutputFormat())*/
    
    val a = tr.filter(_._1._1 == (nFeatures - 1)).map(t => t._1 ->
    t._2.size).reduceGroup(_.mkString(",")).collect()
    val b = tr.filter(_._1._1 == 10).map(t => t._1 ->
    t._2.size).reduceGroup(_.mkString(",")).collect()
    val c = tr.filter(_._1._1 == 12).map(t => t._1 ->
    t._2.size).reduceGroup(_.mkString(",")).collect()

    println("ycol values: " + a)
    println("First x values: " + b)
    println("First y values: " + c)
    
    //env.execute()
  }
}
