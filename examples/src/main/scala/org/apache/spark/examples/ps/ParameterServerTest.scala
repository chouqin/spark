package org.apache.spark.examples.ps

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ps.{PSClient, PSContext}

import scala.util.Random

/**
 * A test example for ps.
 * Created by genmao.ygm on 15-3-19.
 */
object ParameterServerTest {

  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf())
    val pssc = new PSContext(sc)
    val model = sc.parallelize(0 to 7)
      .mapPartitions{ iter =>
      val random = new Random(System.currentTimeMillis().toInt)
      iter.map { e => (e.toString, Array(random.nextDouble(), random.nextDouble()))}
    }

    val trainData = sc.parallelize(0 to 100, 4)
      .map(e => ((e % 8).toString, e.toDouble))

    println("number of model: " + model.count())
    println("number of train data: " + trainData.count())

    pssc.loadPSModel(model)

    trainData.runWithPS(2, process)

    val res = pssc.downloadPSModel(Array("4"), 1)
    res.foreach(e => println(e.mkString(",")))
  }

  def process(arr: Array[(String, Double)], psClient: PSClient): Unit = {
    for(i <- 0 until 10) {
      arr.foreach(e => {
        val index = e._1
        val point = e._2
        val row = psClient.get(index)
        val loss = row.map(_ + point)
        val rowStr = row.mkString(",")
        val lossStr = loss.mkString(",")
        println(s"index $index, point $point, row $rowStr, loss $lossStr")
        psClient.update(index, loss)
      })
      psClient.clock()
    }
  }
}
