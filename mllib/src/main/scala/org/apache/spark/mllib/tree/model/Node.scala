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

package org.apache.spark.mllib.tree.model

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.Logging
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext.doubleRDDToDoubleRDDFunctions

/**
 * :: DeveloperApi ::
 * Node in a decision tree
 * @param id integer node id
 * @param predict predicted value at the node
 * @param isLeaf whether the leaf is a node
 * @param split split to calculate left and right nodes
 * @param leftNode  left child
 * @param rightNode right child
 * @param stats information gain stats
 */
@DeveloperApi
class Node (
    val id: Int,
    val predict: Double,
    var isLeaf: Boolean,
    val split: Option[Split],
    var leftNode: Option[Node],
    var rightNode: Option[Node],
    val stats: Option[InformationGainStats]) extends Serializable with Logging {

  override def toString = "id = " + id + ", isLeaf = " + isLeaf + ", predict = " + predict + ", " +
    "split = " + split + ", stats = " + stats

  /**
   * build the left node and right nodes if not leaf
   * @param nodes array of nodes
   */
  def build(nodes: Array[Node]): Unit = {

    logDebug("building node " + id + " at level " +
      (scala.math.log(id + 1)/scala.math.log(2)).toInt )
    logDebug("id = " + id + ", split = " + split)
    logDebug("stats = " + stats)
    logDebug("predict = " + predict)
    if (!isLeaf) {
      val leftNodeIndex = id * 2 + 1
      val rightNodeIndex = id * 2 + 2
      leftNode = Some(nodes(leftNodeIndex))
      rightNode = Some(nodes(rightNodeIndex))
      leftNode.get.build(nodes)
      rightNode.get.build(nodes)
    }
  }


  /**
   * predict value if node is not leaf
   * @param features feature value
   * @return predicted value
   */
  def predict(features: Vector) : Double = {
    if (isLeaf) {
      predict
    } else{
      if (splitLeft(features)) {
        leftNode.get.predict(features)
      } else {
        rightNode.get.predict(features)
      }
    }
  }

  /**
   *
   * @param validationInput
   * @param isClassification
   * @param threshold
   */
  def prune(validationInput: RDD[(Double, Double)],
            isClassification: Boolean, threshold: Double=0.1) : Unit = {
    if (!isLeaf) {
      if (predictErrorLeaf(validationInput, isClassification) <=
        predictError(validationInput, isClassification) + threshold) {
        // if this node as leaf has less error, remove left and right branches
        isLeaf = true
        leftNode = None
        rightNode = None
      } else {
        leftNode.get.prune(validationInput, isClassification, threshold)
        rightNode.get.prune(validationInput, isClassification, threshold)
      }
    }
  }

  private def splitLeft(features: Vector) : Boolean = {
    if (split.get.featureType == Continuous) {
      features(split.get.feature) <= split.get.threshold
    } else {
      split.get.categories.contains(features(split.get.feature))
    }
  }

  private def predictErrorLeaf(input: RDD[(Double, Double)], isClassification: Boolean) : Double = {
    val predictAndLabels = input.map(p => (predict, p._2))
    predictError(predictAndLabels, isClassification)
  }

  private def predictError(input: RDD[(Double, Double)], isClassification: Boolean) : Double = {
    if (isClassification) {
      1 - accuracyScore(input)
    } else {
      meanSquaredError(input)
    }
  }

  /**
   * Calculates the classifier accuracy.
   */
  private def accuracyScore(data: RDD[(Double, Double)]): Double = {
    val correctCount = data.filter(y => y._1 == y._2).count()
    val count = data.count()
    correctCount.toDouble / count
  }

  /**
   * Calculates the mean squared error for regression.
   */
  private def meanSquaredError(data: RDD[(Double, Double)]): Double = {
    data.map { y =>
      val err = y._1 - y._2
      err * err
    }.mean()
  }


  /**
   * Get the number of nodes in tree below this node, including leaf nodes.
   * E.g., if this is a leaf, returns 0.  If both children are leaves, returns 2.
   */
  private[tree] def numDescendants: Int = {
    if (isLeaf) {
      0
    } else {
      2 + leftNode.get.numDescendants + rightNode.get.numDescendants
    }
  }

  /**
   * Get depth of tree from this node.
   * E.g.: Depth 0 means this is a leaf node.
   */
  private[tree] def subtreeDepth: Int = {
    if (isLeaf) {
      0
    } else {
      1 + math.max(leftNode.get.subtreeDepth, rightNode.get.subtreeDepth)
    }
  }

  /**
   * Recursive print function.
   * @param indentFactor  The number of spaces to add to each level of indentation.
   */
  private[tree] def subtreeToString(indentFactor: Int = 0): String = {

    def splitToString(split: Split, left: Boolean): String = {
      split.featureType match {
        case Continuous => if (left) {
          s"(feature ${split.feature} <= ${split.threshold})"
        } else {
          s"(feature ${split.feature} > ${split.threshold})"
        }
        case Categorical => if (left) {
          s"(feature ${split.feature} in ${split.categories.mkString("{",",","}")})"
        } else {
          s"(feature ${split.feature} not in ${split.categories.mkString("{",",","}")})"
        }
      }
    }
    val prefix: String = " " * indentFactor
    if (isLeaf) {
      prefix + s"Predict: $predict\n"
    } else {
      prefix + s"If ${splitToString(split.get, left=true)}\n" +
        leftNode.get.subtreeToString(indentFactor + 1) +
        prefix + s"Else ${splitToString(split.get, left=false)}\n" +
        rightNode.get.subtreeToString(indentFactor + 1)
    }
  }

}
