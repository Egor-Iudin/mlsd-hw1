import os
import sys
import logging

sys.path.append(".")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainALS")

minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "")
minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "")
minio_endpoint = os.environ.get("MINIO_ENDPOINT", "")
bucket_name = os.environ.get("BUCKET_NAME", "movielens")
train_data_key = os.environ.get("TRAIN_DATA_KEY", "train/ratings_train.csv")
model_output_key = os.environ.get("MODEL_OUTPUT_KEY", "models/als_model")
random_seed = os.environ.get("RANDOM_SEED", "2024")


logger.info("Starting Spark session for prediction job.")

spark = SparkSession.builder.appName("TrainALS").getOrCreate()

hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.access.key", minio_access_key)
hadoop_conf.set("fs.s3a.secret.key", minio_secret_key)
hadoop_conf.set("fs.s3a.endpoint", minio_endpoint)
hadoop_conf.set("fs.s3a.connection.ssl.enabled", "true")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.attempts.maximum", "1")
hadoop_conf.set("fs.s3a.connection.establish.timeout", "5000")
hadoop_conf.set("fs.s3a.connection.timeout", "10000")

spark.sparkContext.setLogLevel("WARN")

data_path = f"s3a://{bucket_name}/{train_data_key}"
logger.info(f"Reading training data from {data_path}")
ratings = spark.read.csv(
    data_path,
    header=True,
    inferSchema=True,
)
train, val = ratings.randomSplit([0.8, 0.2], seed=random_seed)

logger.info("Fitting ALS model...")
als = ALS(
    maxIter=5,
    regParam=0.01,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
)
model = als.fit(train)
logger.info("Model training completed.")

logger.info("Evaluating model on validation set...")
predictions = model.transform(val)
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction",
)
logger.info(f"Validation RMSE: {evaluator.evaluate(predictions)}")

model_path = f"s3a://{bucket_name}/{model_output_key}"
logger.info(f"Saving model to {model_path}")
model.write().overwrite().save(model_path)
logger.info("Model saved successfully.")

spark.stop()

logger.info("Spark session stopped.")
