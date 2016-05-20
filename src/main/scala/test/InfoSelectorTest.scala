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
import org.apache.flink.ml.pipeline.Transformer
import org.apache.flink.ml.preprocessing.StandardScaler
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector

/**
 * This example implements a basic Linear Regression  to solve the y = theta0 + theta1*x problem
 * using batch gradient descent algorithm.
 *
 * Linear Regression with BGD(batch gradient descent) algorithm is an iterative algorithm and
 * works as follows:
 *
 * Giving a data set and target set, the BGD try to find out the best parameters for the data set
 * to fit the target set.
 * In each iteration, the algorithm computes the gradient of the cost function and use it to
 * update all the parameters.
 * The algorithm terminates after a fixed number of iterations (as in this implementation).
 * With enough iteration, the algorithm can minimize the cost function and find the best parameters
 * This is the Wikipedia entry for the
 * [[http://en.wikipedia.org/wiki/Linear_regression Linear regression]] and
 * [[http://en.wikipedia.org/wiki/Gradient_descent Gradient descent algorithm]].
 *
 * This implementation works on one-dimensional data and finds the best two-dimensional theta to
 * fit the target.
 *
 * Input files are plain text files and must be formatted as follows:
 *
 *  - Data points are represented as two double values separated by a blank character. The first
 *    one represent the X(the training data) and the second represent the Y(target). Data points are
 *    separated by newline characters.
 *    For example `"-0.02 -0.04\n5.3 10.6\n"`gives two data points
 *    (x=-0.02, y=-0.04) and (x=5.3, y=10.6).
 *
 * This example shows how to use:
 *
 *  - Bulk iterations
 *  - Broadcast variables in bulk iterations
 */
object InfoSelectorTest {

  def main(args: Array[String]) {

    val params: ParameterTool = ParameterTool.fromArgs(args)
    println("Usage: LinearRegression --input <path> --output <path> --iterations <n>")

    // set up execution environment
    val env = ExecutionEnvironment.getExecutionEnvironment

    // make parameters available in the web interface
    val conf = env.getConfig
    
    conf.setGlobalJobParameters(params)
    conf.registerKryoType(classOf[breeze.linalg.SparseVector[_]])
    conf.registerKryoType(classOf[breeze.linalg.Vector[_]])
    conf.registerKryoType(classOf[breeze.linalg.DenseVector[_]])
    conf.registerKryoType(classOf[breeze.linalg.DenseMatrix[_]])   
    conf.registerKryoType(classOf[(Int, breeze.linalg.SparseVector[_])])
    
    //conf.disableForceKryo()
    
    val header = "/home/sramirez/datasets/ECBDL14/ECBDL14.header"
    val typeConversion = KeelParser.parseHeaderFile(env, header) 
    val lines = env.readTextFile("/home/sramirez/datasets/ECBDL14/subSetROS.data")
    val training = lines.map(line => KeelParser.parseLabeledPoint(typeConversion, line))  
    
    val data = List(LabeledVector(1.0, DenseVector(1.0, 2.0)),
      LabeledVector(2.0, DenseVector(2.0, 3.0)),
      LabeledVector(3.0, DenseVector(3.0, 4.0)))
    val training2 = MLUtils.readLibSVM(env, "/home/sramirez/datasets/a1a.txt")
    
    val training3 = env.fromCollection(data)
    val training4 = env.readTextFile("/home/sramirez/datasets/ECBDL14/disc/subSetROS_disc_data_headers.csv")
        .filter(!_.startsWith("separation"))
        .map(line => KeelParser.parseLabeledPoint(typeConversion, line)) 
    
    val disc = FrequencyDiscretizer().setNBuckets(10)  
    val selector = InfoSelector().setNFeatures(15)
    //val selector = StandardScaler()
    
    // Construct pipeline of standard scaler, polynomial features and multiple linear regression
    //val pipeline = disc.chainTransformer(selector)
    
    // Train pipeline
    //disc.fit(training)
    selector.fit(training4)
    val output = selector.transform(training4)
    println("Result: " + output.collect().take(10).mkString("\n"))
    //output.writeAsText("/home/sramirez/datasets/ECBDL14/asd.data")
    //val output = pipeline.transform(training2, new ParameterMap())
    //println("Selected features: " + disc.splits.get.apply(0).mkString(","))
    //println("Selected features: " + selector.selectedFeatures.get.mkString(","))
  }
}
