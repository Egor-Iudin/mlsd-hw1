import os
import sys
import logging

sys.path.append(".")

import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictALS")

minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "")
minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "")
minio_endpoint = os.environ.get("MINIO_ENDPOINT", "")
bucket_name = os.environ.get("BUCKET_NAME", "movielens")

logger.info("Starting Spark session for prediction job.")

spark = SparkSession.builder.appName("PredictALS").getOrCreate()

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

model_path = f"s3a://{bucket_name}/models/als_model"
test_data_path = f"s3a://{bucket_name}/test/ratings_test.csv"
output_path = f"s3a://{bucket_name}/predictions/raw_predictions"

logger.info(f"Loading ALS model from {model_path}")
model = ALSModel.load(model_path)

logger.info(f"Reading test data from {test_data_path}")
test = spark.read.csv(test_data_path, header=True, inferSchema=True)

logger.info("Generating predictions...")
predictions = model.transform(test)


@pandas_udf("double")
def adjust_prediction_udf(cols: pd.Series) -> pd.Series:
    return cols.clip(0, 5)


logger.info("Adjusting predictions to the [0,5] range.")
predictions = predictions.withColumn(
    "prediction_adj",
    adjust_prediction_udf(col("prediction")),
)

logger.info("Checking prediction results:")
logger.info(f"Number of predicted rows: {predictions.count()}")
logger.info("Sample predictions:")
predictions.show(5)

logger.info(f"Writing predictions to {output_path}")
predictions.coalesce(1).write.csv(
    output_path,
    header=True,
    mode="overwrite",
)
logger.info("Predictions successfully saved.")

spark.stop()

logger.info("Spark session stopped.")
