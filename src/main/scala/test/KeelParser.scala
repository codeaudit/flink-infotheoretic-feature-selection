package test

import keel.Dataset._
import java.util.ArrayList
import collection.JavaConverters._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.ml.math.DenseVector

object KeelParser {
  
	// Only call once (KEEL append lines to the header)
	def parseHeaderFile (env: ExecutionEnvironment, file: String): Array[Map[String, Double]] = {
  	  val header = scala.io.Source.fromFile(file).getLines()
  	  // Java code classes
  	  var arr: ArrayList[String] = new ArrayList[String]
  	  // Important to collect and work with arrays instead of RDD's
  	  for(x <- header) arr.add(x)
      
  	  
  	  Attributes.clearAll()
  	  new InstanceSet().parseHeaderFromString(arr, true)
  	  
  	  val conv = new Array[Map[String, Double]](Attributes.getNumAttributes)
  	  for(i <- 0 until Attributes.getNumAttributes) {
  		  conv(i) = Map()
  		  if(Attributes.getAttribute(i).getType == Attribute.NOMINAL){
  			  val values = Attributes.getAttribute(i)
  					  .getNominalValuesList()
  					  .asInstanceOf[java.util.Vector[String]]
  			  val gen = for (j <- 0 until values.size) yield (values.get(j) -> j.toDouble) 
			    conv(i) = gen.toMap
  		  } else {
          val min = Attributes.getAttribute(i).getMinAttribute()
          conv(i) = Map("min" -> min) 
        } 
  	  }
  	  //conv(Attributes.getNumAttributes - 1) = Map()      
  	  conv
  	}
  
	def parseLabeledPoint (conv: Array[Map[String, Double]], str: String): LabeledVector = {
	  
		val tokens = str split ","
		require(tokens.length == conv.length)
		
		val arr = (conv, tokens).zipped.map{(c, elem) => 
      c.getOrElse(elem, elem.toDouble)
            /*c.get("min") match {
              case Some(min) => elem.toDouble - min
              case None => c.getOrElse(elem, elem.toDouble)
            } */          
        }
        
		val features = arr.slice(0, arr.length - 1)
		val label = arr.last
		
		new LabeledVector(label, new DenseVector(features))
	}
}