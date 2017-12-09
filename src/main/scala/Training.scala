import TransformUtils.{mirror, rotate}
import org.apache.spark.ml.classification.{GBTClassifier, MultilayerPerceptronClassifier, _}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{Dataset, SparkSession}

object Training {

  case class TrainRecord(label: Double, features: Vector)

  // convert each input line to TrainRecord
  def toTrainRecord(line: String): TrainRecord = {
    val res = line.split(",").map(_.toDouble)
    TrainRecord(res.last, Vectors.dense(res.init))
  }

  // use MulticlassClassificationEvaluator for evaluation each parameter set
  def getEvaluator(): MulticlassClassificationEvaluator = {
    new MulticlassClassificationEvaluator()
      .setPredictionCol("prediction")
      .setLabelCol("label")
      .setMetricName("accuracy")
  }

  // take a set of parameters, train model on each parameter
  // evaluate them on validatingData and return the one with highest accuracy
  def trainValidation(paramGrid: Array[ParamMap],
                      trainingData: Dataset[TrainRecord],
                      validatingData: Dataset[TrainRecord],
                      estimator: Estimator[_]): (Model[_], Double) = {
    // models of different parameters
    trainingData.persist()
    val models = paramGrid.map(paramMap =>
      estimator.fit(trainingData, paramMap).asInstanceOf[Model[_]])
    trainingData.unpersist()
    // evaluate each model on validation data
    val matrics = models.zip(paramGrid).map {
      case (model, paramMap) =>
        getEvaluator().evaluate(model.transform(validatingData, paramMap))
    }
    // return the best model based on accuracy
    models.zip(matrics).maxBy(_._2)
  }

  def main(args: Array[String]): Unit = {
    // Prepare constants
    val TRAINING_RATIO = 0.33

    // Prepare args
    if (args.length < 2) {
      System.exit(1)
    }
    val inputPath = args(0)
    val outputPath = args(1)

    // Prepare Spark
    val spark = SparkSession.builder().appName("NCTracer TrainingV2").getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    // evaluate on testing data and save the accuracy - only for debugging and local testing
    def evaluateTestData(model: Model[_],
                         modelName: String,
                         testingData: Dataset[TrainRecord]): Unit = {
      val predictionData = model.transform(testingData)
      val statistic = List(
        predictionData.filter(r => r.getAs[Double]("label") != r.getAs[Double]("prediction") && r.getAs[Double]("label") == 1.0).count,
        predictionData.filter(r => r.getAs[Double]("label") == 1.0).count,
        predictionData.filter(r => r.getAs[Double]("label") != r.getAs[Double]("prediction") && r.getAs[Double]("label") == 0.0).count,
        predictionData.filter(r => r.getAs[Double]("label") == 0.0).count)
      sc.parallelize(
        Seq(getEvaluator().evaluate(predictionData), statistic, model.explainParams()),
        1)
        .saveAsTextFile(outputPath + "/res/" + modelName)
    }

    // Prepare training data
    val trainingData = sc.textFile(inputPath + "/training")
      .map(toTrainRecord)
      .toDS().sample(false, TRAINING_RATIO).repartition(500)

    // Prepare validating data
    val validatingData = sc.textFile(inputPath + "/validation")
      .map(toTrainRecord)
      .toDS()

    // Prepare testing data - only for debugging and local testing
    //    val testingData = sc.textFile(inputPath + "/testing", 200)
    //      .map(toTrainRecord)
    //      .toDS()

    // Train each model
    // RF - random forest, BY - naive bayers, LR - logistic regression
    val jobs = Seq("RF", "BY", "LR")

    jobs.par.map(job => {
      val modelName = job
      val metaData: (Estimator[_], Array[ParamMap], Dataset[TrainRecord]) =
        if (modelName == "DT") {
          val dt = new DecisionTreeClassifier()
          // parameter sets for decision tree
          val paramGrid = new ParamGridBuilder()
            .addGrid(dt.maxBins, Array(30))
            .addGrid(dt.maxDepth, Array(30))
            .addGrid(dt.thresholds, Array(Array(0.02, 0.98)))
            .build()
          (dt, paramGrid, trainingData)
        } else if (modelName == "LR") {
          // transform the data so that each model train on different data
          val transformedTrainingData = trainingData.map(r =>
            TrainRecord(r.label, mirror(r.features, 0)))
          val lr = new LogisticRegression()
          // parameter sets for logistic regression
          val paramGrid = new ParamGridBuilder()
            .addGrid(lr.maxIter, Array(8))
            .addGrid(lr.regParam, Array(0.1))
            .addGrid(lr.threshold, Array(0.18))
            .build()
          (lr, paramGrid, transformedTrainingData)
        } else if (modelName == "SVC") {
          // transform the data so that each model train on different data
          val transformedTrainingData = trainingData.map(r =>
            TrainRecord(r.label, rotate(r.features, 90)))
          val svc = new LinearSVC()
          // parameter sets for SVC
          val paramGrid = new ParamGridBuilder()
            .addGrid(svc.maxIter, Array(10, 15, 20))
            .addGrid(svc.regParam, Array(0.1))
            .addGrid(svc.threshold, Array(0.005))
            .build()
          (svc, paramGrid, transformedTrainingData)
        } else if (modelName == "GBT") {
          // transform the data so that each model train on different data
          val transformedTrainingData = trainingData.map(r =>
            TrainRecord(r.label, mirror(rotate(r.features, 90), 90)))
          val gbt = new GBTClassifier()
          // parameter sets for gradient boosting
          val paramGrid = new ParamGridBuilder()
            .addGrid(gbt.maxIter, Array(90))
            .build()
          (gbt, paramGrid, transformedTrainingData)
        } else if (modelName == "PMC") {
          // transform the data so that each model train on different data
          val transformedTrainingData = trainingData.map(r =>
            TrainRecord(r.label, rotate(r.features, 180)))
          val mpc = new MultilayerPerceptronClassifier()
          // parameter sets for PMC
          val paramGrid = new ParamGridBuilder()
            .addGrid(mpc.layers, Array(Array(3087, 50, 10, 2)))
            .addGrid(mpc.seed, Array(1234L))
            .addGrid(mpc.blockSize, Array(64, 128, 256))
            .addGrid(mpc.maxIter, Array(50, 75, 100, 125, 150))
            .build()
          (mpc, paramGrid, transformedTrainingData)
        } else if (modelName == "BY") {
          // transform the data so that each model train on different data
          val transformedTrainingData = trainingData.map(r =>
            TrainRecord(r.label, mirror(rotate(r.features, 180), 180)))
          val by = new NaiveBayes()
          // parameter sets for naive bayers
          val paramGrid = new ParamGridBuilder()
            .addGrid(by.smoothing, Array(1.0))
            .addGrid(by.thresholds, Array(Array(0.0, 1.0)))
            .build()
          (by, paramGrid, transformedTrainingData)
        } else {
          // RF, random forest
          // transform the data so that each model train on different data
          val transformedTrainingData = trainingData.map(r =>
            TrainRecord(r.label, rotate(r.features, 270)))
          val rf = new RandomForestClassifier()
          // parameter sets for random forest
          val paramGrid = new ParamGridBuilder()
            .addGrid(rf.maxBins, Array(32))
            .addGrid(rf.numTrees, Array(20))
            .addGrid(rf.maxDepth, Array(14))
            .addGrid(rf.thresholds, Array(Array(0.5, 0.5)))
            .build()
          (rf, paramGrid, transformedTrainingData)
        }
      val bestModel = trainValidation(metaData._2, metaData._3, validatingData, metaData._1)
      // evaluateTestData(bestModel._1, modelName, testingData)
      bestModel._1.asInstanceOf[MLWritable].save(outputPath + "/models/" + modelName)
    })
  }
}
