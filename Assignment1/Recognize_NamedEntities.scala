// Databricks notebook source
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.util.Benchmark
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._


// COMMAND ----------

val pipeline = PretrainedPipeline("recognize_entities_dl", "en")

// COMMAND ----------

val filePath = "/FileStore/tables/SherlockHolmes.txt"

// COMMAND ----------

val df = spark.read.option("header", false).textFile(filePath).toDF("text")

// COMMAND ----------

val data = pipeline.transform(df).select("entities.result").filter(size($"result") > 0)

// COMMAND ----------

display(data)

// COMMAND ----------

data.printSchema()

// COMMAND ----------

val rdd = data.withColumn("result", explode(col("result"))).select("result").rdd
rdd.collect()

// COMMAND ----------

val entityCount = rdd.map(word => (word,1)).reduceByKey((x,y) => x+y).sortBy(-_._2)


// COMMAND ----------

entityCount.collect()
