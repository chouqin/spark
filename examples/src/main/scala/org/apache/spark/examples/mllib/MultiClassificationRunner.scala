package org.apache.spark.examples.mllib

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.classification.{MultiClassification, SVMWithSGD}

object MultiClassificationRunner {
  def loadIrisData(sc: SparkContext, filename:String): RDD[LabeledPoint] = {
    val nameToLabels = Map[String, Double](
      "Iris-setosa" -> 0.0,
      "Iris-versicolor" -> 1.0,
      "Iris-virginica" -> 2.0
    )
    sc.textFile(filename).map(line => {
      val items = line.split(",")
      val label = nameToLabels(items.last)
      val features = new DenseVector(items.slice(0, items.length-1).map(_.toDouble))
      new LabeledPoint(label, features)
    })
  }

  def main(args: Array[String]) {
    val filename = args(0)
    val conf = new SparkConf().setAppName("MultiClassificationRunner")
    val sc = new SparkContext(conf)

    val input = loadIrisData(sc, filename)

    runMultiSVM(input)

  }

  // 训练以SVM为基础的多分类器
  def runMultiSVM(input: RDD[LabeledPoint]) {
    val numIterations = 100
    val stepSize = 1.0
    val regParam = 1.0
    val miniBatchFraction = 1.0
    val baseClassifier = new SVMWithSGD(stepSize, numIterations, regParam, miniBatchFraction)

    val model = new MultiClassification(baseClassifier, 3).run(input)
    val result = input.map(p => {
      (p.label, model.predict(p.features), model.predictProb(p.features))
    }).collect()

    result.foreach(t => {
      println(t._1, t._2, t._3)
    })

    val error = result.filter(t => t._1 != t._2).length
    println(error, result.length)
  }
}
