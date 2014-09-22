package org.apache.spark.mllib.classification

import org.apache.spark.mllib.regression.{LabeledPoint, GeneralizedLinearAlgorithm}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector
import scala.reflect.ClassTag

class MultiClassficiationModel[M<: ClassificationModel]
    (val baseEstimators: Array[M]) {
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

class MultiClassification[M<: ClassificationModel: ClassTag] (
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
      model.clearThreshold()
      estimators(cls) = model

      cls += 1
    }

   new MultiClassficiationModel[M](estimators)

  }

}
