import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession

object Prediction {

  // index each pixel so that we can join each model' s result and get the votes
  case class IndexedRecord(label: Double, features: Vector, index: Long)

  case class PredictionRecord(label: Double, prediction: Double)

  // map each model name to their corresponding model class
  val loadMapping = Map(
    "DT" -> DecisionTreeClassificationModel,
    "LR" -> LogisticRegressionModel,
    "SVC" -> LinearSVCModel,
    "GBT" -> GBTClassificationModel,
    "PMC" -> MultilayerPerceptronClassificationModel,
    "RF" -> RandomForestClassificationModel,
    "BY" -> NaiveBayesModel)

  // convert each line to IndexedRecord
  def toIndexedRecord(record: (String, Long)): IndexedRecord = {
    val (line, index) = record
    val res = line.split(",")
    IndexedRecord(0.0, Vectors.dense(res.init.map(_.toDouble)), index)
  }

  def main(args: Array[String]): Unit = {
    // path of input (image file)
    val inputPath = args(0)
    // path of output (models and prediction result)
    val outputPath = args(1)
    val modelInputPath = outputPath + "/models/"

    // Prepare Spark
    val spark = SparkSession.builder().appName("NCTracer Prediction").getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    // read input data and convert to Dataset of IndexedRecord
    val predictionData = sc.textFile(inputPath + "/test", 400)
      .zipWithIndex()
      .map(toIndexedRecord)
      .toDS()

    // models that will be used for voting
    val modelNames = Seq("LR", "BY", "RF")
    val results = modelNames.map { modelName =>
      val model = loadMapping(modelName).load(modelInputPath + modelName)
      val result = model.transform(predictionData)
      result.withColumnRenamed("prediction", "prediction" + modelName)
    }.reduce((d1, d2) => d1.join(d2, "index")) // aggregate the results of each model
      .map { r =>
      val votes = modelNames.map(modelName => r.getAs[Double]("prediction" + modelName)).reduce(_ + _)
      // if half of the models vote for foreground, then predict the pixel as foreground
      val predictionLabel = if (votes > modelNames.size / 2) 1.0 else 0.0
      PredictionRecord(r.getAs[Double]("label"), predictionLabel)
    }.persist()
    results.map(r => r.prediction.toInt).repartition(1).rdd.saveAsTextFile(outputPath + "/prediction_result")
  }
}