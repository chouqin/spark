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

package org.apache.spark.mllib.classification

import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint, GeneralizedLinearAlgorithm}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector
import scala.reflect.ClassTag

/**
 *
 * @param baseEstimators: 基础的二分类器模型，每一个类一个
 * @tparam M: 基础二分类器模型的类型，必须实现`ClassificationWithProbModel`这个trait，
 *            也就是要实现`predictProb`这个方法
 */
class MultiClassficiationModel[M<: ClassificationWithProbModel]
    (val baseEstimators: Array[M]) extends Serializable {

  /**
   * 预测样本属于哪一个类
   * @param x 样本的特征向量
   * @return
   */
  def predict(x: Vector): Int = {
    val probs = predictProb(x)
    probs.zipWithIndex.maxBy(_._1)._2
  }

  /**
   * 预测样本属于每一个类的概率
   * @param x 样本的特征向量
   * @return
   */
  def predictProb(x: Vector): Array[Double] = {
    baseEstimators.map { e =>
      e.predictProb(x)
    }
  }
}

/**
 *
 * @param baseClassifier 基础二分类算法，能够根据训练数据`input`，训练出一个基础二分类器模型
 *                       基础分类器算法必须实现`run`方法，其接受一个`input`作为输入，返回一个二分类器模型
 * @param numClasses 类型的个数
 * @tparam M 基础分类器模型的类型
 */
class MultiClassification[M<: GeneralizedLinearModel with ClassificationWithProbModel: ClassTag] (
    val baseClassifier: GeneralizedLinearAlgorithm[M],
    val numClasses: Int) extends Serializable {

  /**
   * 根据输入数据训练多分类模型`MultiClassficiationModel`，用于多分类预测
   * @param input 输入数据，label的范围从0到numClasses-1
   * @return
   */
  def run(input: RDD[LabeledPoint]): MultiClassficiationModel[M] = {
    input.cache()

    val estimators = new Array[M](numClasses)
    var cls = 0
    while (cls < numClasses) {
      val currentLabel = cls
      val binaryInput = input.map(p => {
        val label = if (p.label == currentLabel) 1.0 else 0.0
        LabeledPoint(label, p.features)
      })
      val model: M = baseClassifier.run(binaryInput)
      estimators(cls) = model

      cls += 1
    }

    new MultiClassficiationModel[M](estimators)

  }

}
