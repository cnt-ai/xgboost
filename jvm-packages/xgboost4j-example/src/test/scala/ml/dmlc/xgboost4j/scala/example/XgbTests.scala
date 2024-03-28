/*
 Copyright (c) 2023 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package ml.dmlc.xgboost4j.scala.example

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost}
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import java.io.File
import scala.collection.mutable

class XgbTests extends SparkTest {
  val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("rawPrediction")
    .setMetricName("areaUnderROC")

  def defaultXgbParams(params: mutable.HashMap[String, Any]): Unit = {
    params += "booster" -> "gbtree"
    params += "eta" -> 1.0
    params += "gamma" -> 1.0
    params += "max_depth" -> 8
    params += "seed" -> 0
    params += "min_child_weight" -> 0
    params += "missing" -> 0f
    params += "verbosity" -> 2
    params += "objective" -> "binary:logistic"
    params += "tree_method" -> "hist"
    params += "eval_metric" -> "auc"
    params += "dump_format" -> "json"
  }

  val params = new mutable.HashMap[String, Any]()
  defaultXgbParams(params)

  override def initSparkConf(): SparkConf = super.initSparkConf().set("spark.task.cpus", "2")

  "spark xgb" should "work with a9a dataset" in {
    val trainInput = spark.read.format("libsvm").load("../demo/data/a9a.train")
    val testInput = spark.read.format("libsvm").load("../demo/data/a9a.test")

    // params += "num_workers" -> 2
    params += "timeout_request_workers" -> 60000L
    params += "eval_sets" -> Map("test" -> testInput)

    val xgbClassifier = new XGBoostClassifier(params.toMap).setMissing(0.0f)
    val xgbModel = xgbClassifier.fit(trainInput)
    val testAUC = evaluator.evaluate(xgbModel.transform(testInput))
    println(s"Test AUC: $testAUC")
  }

  "xgb" should "work with a9a dataset" in {
    val trainMax = new DMatrix("../demo/data/a9a.train?format=libsvm")
    val testMax = new DMatrix("../demo/data/a9a.test?format=libsvm")

    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMax
    watches += "test" -> testMax

    val round = 3
    // train a model
    val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)
    // predict
    val predicts = booster.predict(testMax)
    // save model to model path
    val file = new File("./model/a9a")
    if (!file.exists()) {
      file.mkdirs()
    }
    booster.saveModel(file.getAbsolutePath + "/xgb.model.json")
    // save dmatrix into binary buffer
    testMax.saveBinary(file.getAbsolutePath + "/dtest.buffer")

  }

  "xgb" should "work with agaricus dataset" in {
    val trainMax = new DMatrix("../demo/data/agaricus.txt.train?format=libsvm")
    val testMax = new DMatrix("../demo/data/agaricus.txt.test?format=libsvm")

    val watches = new mutable.HashMap[String, DMatrix]
    watches += "train" -> trainMax
    watches += "test" -> testMax

    val round = 3
    // train a model
    val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)
    // predict
    val predicts = booster.predict(testMax)
    // save model to model path
    val file = new File("./model/agaricus")
    if (!file.exists()) {
      file.mkdirs()
    }
    booster.saveModel(file.getAbsolutePath + "/xgb.model.json")
    // dump model with feature map
    // val modelInfos = booster.getModelDump("../demo/data/featmap.txt")

    // save dmatrix into binary buffer
    testMax.saveBinary(file.getAbsolutePath + "/dtest.buffer")

    // reload model and data
    val booster2 = XGBoost.loadModel(file.getAbsolutePath + "/xgb.model.json")
    val testMax2 = new DMatrix(file.getAbsolutePath + "/dtest.buffer")
    val predicts2 = booster2.predict(testMax2)
  }

}
