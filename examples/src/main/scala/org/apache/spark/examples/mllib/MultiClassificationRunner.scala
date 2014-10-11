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

package org.apache.spark.examples.mllib

import org.apache.spark.mllibnew.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllibnew.linalg.DenseVector
import org.apache.spark.mllibnew.classification.{LogisticRegressionWithSGD, MultiClassification, SVMWithSGD}

object MultiClassificationRunner {
  def main(args: Array[String]) {
    val filename = args(0)
    val conf = new SparkConf().setAppName("MultiClassificationRunner")
    val sc = new SparkContext(conf)

//    val input = loadIrisData(sc, filename)

//    runMultiSVM(input)
//
//    runMultiLR(input)


    val binaryInput = loadBinaryIrisData(sc, filename)
    runSVM(binaryInput)
  }

  // 加载Iris DataSet作为数据
  def loadIrisData(sc: SparkContext, filename:String): RDD[LabeledPoint] = {

    // 从类型名到label的对应
    val nameToLabels = Map[String, Double](
      "Iris-setosa" -> 0.0,
      "Iris-versicolor" -> 1.0,
      "Iris-virginica" -> 2.0
    )
    sc.textFile(filename).map(line => {
      val items = line.split(",")
      // 最后一项是label，前面的是features
      val label = nameToLabels(items.last)
      val features = new DenseVector(items.slice(0, items.length-1).map(_.toDouble))
      new LabeledPoint(label, features)
    })
  }

  // 加载label为Iris-versicolor和Iris-virginica的数据集，测试二分类器
  def loadBinaryIrisData(sc: SparkContext, filename:String): RDD[LabeledPoint] = {

    // 从类型名到label的对应
    val nameToLabels = Map[String, Double](
      "Iris-setosa" -> 2.0,
      "Iris-versicolor" -> 1.0,
      "Iris-virginica" -> 0.0
    )
    sc.textFile(filename).map(line => {
      val items = line.split(",")
      // 最后一项是label，前面的是features
      val label = nameToLabels(items.last)
      val features = new DenseVector(items.slice(0, items.length-1).map(_.toDouble))
      new LabeledPoint(label, features)
    }).filter(_.label != 2.0)
  }

  def runSVM(input: RDD[LabeledPoint]) {
    // 算法的参数，需要调节
    val numIterations = 100
    val stepSize = 1.0
    val regParam = 1.0
    val miniBatchFraction = 1.0
    // 构建基础分类器
    val baseClassifier = new SVMWithSGD(stepSize, numIterations, regParam, miniBatchFraction)
    val model = baseClassifier.run(input)
    val result = input.map(p => {
      (p.label, model.predict(p.features))
    }).collect()

    result.foreach(t => {
      println(t._1, t._2)
    })

    val error = result.filter(t => t._1 != t._2).length
    println(error, result.length)
  }


  // 训练以SVM为基础的多分类器
  def runMultiSVM(input: RDD[LabeledPoint]) {
    // 算法的参数，需要调节
    val numIterations = 100
    val stepSize = 1.0
    val regParam = 1.0
    val miniBatchFraction = 1.0
    // 构建基础分类器
    val baseClassifier = new SVMWithSGD(stepSize, numIterations, regParam, miniBatchFraction)

    val model = new MultiClassification(baseClassifier, 3).run(input)
    val result = input.map(p => {
      (p.label, model.predict(p.features), model.predictProb(p.features))
    }).collect()

    result.foreach(t => {
      println(t._1, t._2, t._3.mkString(" "))
    })

    val error = result.filter(t => t._1 != t._2).length
    println(error, result.length)
  }

  // 训练以LR为基础的多分类器
  def runMultiLR(input: RDD[LabeledPoint]) {

    // 算法的参数，需要调节
    val numIterations = 100
    val stepSize = 1.0
    val regParam = 1.0
    val miniBatchFraction = 1.0
    // 构建基础分类器
    val baseClassifier = new LogisticRegressionWithSGD(stepSize, numIterations,
      regParam, miniBatchFraction)

    val model = new MultiClassification(baseClassifier, 3).run(input)
    val result = input.map(p => {
      (p.label, model.predict(p.features), model.predictProb(p.features))
    }).collect()

    result.foreach(t => {
      println(t._1, t._2, t._3.mkString(" "))
    })

    val error = result.filter(t => t._1 != t._2).length
    println(error, result.length)
  }
}
