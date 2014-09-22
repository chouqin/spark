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

class MultiClassficiationModel[M<: ClassificationWithProbModel]
    (val baseEstimators: Array[M]) extends Serializable{
  def predict(x: Vector): Int = {
    val probs = predictProb(x)
    probs.zipWithIndex.maxBy(_._1)._2
  }

  def predictProb(x: Vector): Array[Double] = {
    baseEstimators.map { e =>
      e.predict(x)
    }
  }
}

class MultiClassification[M<: GeneralizedLinearModel with ClassificationWithProbModel: ClassTag] (
    val baseClassifier: GeneralizedLinearAlgorithm[M],
    val numClasses: Int) extends Serializable{
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
