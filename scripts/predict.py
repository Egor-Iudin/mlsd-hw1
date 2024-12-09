import os
import sys

sys.path.append(".")


import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf

minio_access_key = "minioaccesskey"
minio_secret_key = "miniosecretkey"
minio_endpoint = "http://minio:9000"

spark = SparkSession.builder.appName("PredictALS").getOrCreate()

spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key", minio_access_key)
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key", minio_secret_key)
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", minio_endpoint)
spark.sparkContext._jsc.hadoopConfiguration().set(
    "fs.s3a.connection.ssl.enabled", "true"
)
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.attempts.maximum", "1")
spark.sparkContext._jsc.hadoopConfiguration().set(
    "fs.s3a.connection.establish.timeout", "5000"
)
spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.connection.timeout", "10000")
spark.sparkContext.setLogLevel("WARN")


model = ALSModel.load("s3a://movielens/models/als_model")
test = spark.read.csv(
    "s3a://movielens/test/ratings_test.csv", header=True, inferSchema=True
)

predictions = model.transform(test)


@pandas_udf("double")
def adjust_prediction_udf(col_s: pd.Series) -> pd.Series:
    return col_s.clip(0, 5)


predictions = predictions.withColumn(
    "prediction_adj",
    adjust_prediction_udf(col("prediction")),
)
predictions.coalesce(1).write.csv(
    "s3a://movielens/predictions/raw_predictions",
    header=True,
    mode="overwrite",
)

spark.stop()
