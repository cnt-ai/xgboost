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

import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec

class SparkTest extends AnyFlatSpec with BeforeAndAfterAll with Serializable {
  lazy implicit val spark: SparkSession = {
    val sparkBuild = SparkSession
      .builder
      .appName(appName)
      .config(initSparkConf())
    if (isLocal) {
      sparkBuild.master("local[*]")
    }
    if (enableHiveSupport) {
      sparkBuild.enableHiveSupport()
    }
    sparkBuild.getOrCreate()
  }

  lazy implicit val sc: SparkContext = spark.sparkContext
  lazy implicit val sqc: SQLContext = spark.sqlContext

  def appName: String = this.getClass.getSimpleName

  def isLocal: Boolean = true

  def enableHiveSupport: Boolean = false

  def initSparkConf(): SparkConf = new SparkConf()

}
