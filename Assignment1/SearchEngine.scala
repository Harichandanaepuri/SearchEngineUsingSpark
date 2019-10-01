// Databricks notebook source
val filePath = "/FileStore/tables/plot_summaries.txt"
val searchFilePath = "/FileStore/tables/searchterms.txt"
var movieMetaDataFilePath = "/FileStore/tables/movie_metadata-ab497.tsv"
val plotSummaries = sc.textFile(filePath)
val searchTerms = sc.textFile(searchFilePath)
val movieMetaData= sc.textFile(movieMetaDataFilePath)

// COMMAND ----------

val summaries = plotSummaries.map(_.toLowerCase).map(line => (line.split("\t")(0).toInt, line.split("\t")(1).trim().split("""\W+"""))).toDF("id","raw")
display(summaries)

// COMMAND ----------

import org.apache.spark.ml.feature.StopWordsRemover
val remover = new StopWordsRemover()
  .setInputCol("raw")
  .setOutputCol("filtered")
val cleanDF = remover.transform(summaries)
val cleanDF1 = cleanDF.select("id","filtered")
cleanDF1.show()

// COMMAND ----------

import scala.collection.mutable.WrappedArray
val summaries_new = cleanDF1.rdd.map(row => (row.getInt(0),row.getSeq(1).mkString(",").split(",")))
summaries_new.take(2)

// COMMAND ----------

val wordSummaries = summaries_new.flatMap(l => l._2.map(word => ((l._1, word), 1)))
wordSummaries.take(2)

// COMMAND ----------

val tF = wordSummaries.reduceByKey((x,y) => x+y)

// COMMAND ----------

tF.collect()

// COMMAND ----------

val totalDocs = summaries.count()

// COMMAND ----------

val termDocFreq = wordSummaries.reduceByKey((x,y) =>x).map(x => (x._1._2,x._2)).reduceByKey((x,y) => x+y)

// COMMAND ----------

val idf = termDocFreq.map(x => (x._1, math.log((totalDocs+1)/(x._2+1))))

// COMMAND ----------

val tfIdfJoin = tF.map(x => (x._1._2, x)).join(idf)

// COMMAND ----------

val tfIdf = tfIdfJoin.map(x => (x._2._1._1._1, (x._1, x._2._1._2 * x._2._2)))

// COMMAND ----------

val movieIdAndName = movieMetaData.map(x => (x.split("\t")(0).toInt,x.split("\t")(2)))

// COMMAND ----------

val textFile = spark.read.textFile("/FileStore/tables/searchterms.txt")
val textFile1 = textFile.collect().toSeq
val y = textFile1.mkString(",").split(",")
for(myString <- y) {
   println("Search results for search :"+myString)
   val size = myString.split(" ").size
 if(size >= 2)
  {
    val searchwords = sc.parallelize(myString.toLowerCase().split(" ").map(x=> (x,1)))
    val searchTF = searchwords.reduceByKey((x,y) => x+y)
    val searchTF_IDF = searchTF.join(idf).map(x => (x._1, x._2._1 * x._2._2))
    val searchSquareRoot = math.sqrt(searchTF_IDF.fold(("", 0))((x, y) => (x._1, x._2 + y._2 * y._2))._2)
    val searchDocJoin = tfIdf.map(x => (x._2._1, (x._1, x._2._2))).join(searchTF_IDF).map(x => (x._2._1._1, (x._2._1._2 * x._2._2, x._2._1._2 * x._2._1._2))).foldByKey((0, 0))((x, y) => (x._1 + y._1, x._2 + y._2)).map(x => (x._1, x._2._1/(math.sqrt(x._2._2) * searchSquareRoot)))
    val searchResults = searchDocJoin.join(movieIdAndName)
    val sortedResults = searchResults.sortBy(-_._2._1).map(_._2._2).take(10)
        sortedResults.foreach(println)
  println("------------------------------------------------------")
  }
  else if(size < 2) {
   val top10 = tfIdf.filter(_._2._1 == myString).map(x =>(x._2._2, (x._2._1,x._1))).sortByKey(false).take(10).map(l => (l._2._2,l._1))
  for(movie <- top10) {
  println(movieIdAndName.filter(_._1==movie._1).values.collect().mkString(" "))  
  }
  println("------------------------------------------------------")
  }
}
