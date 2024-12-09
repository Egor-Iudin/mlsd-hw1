import os
import sys

sys.path.append(".")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

minio_access_key = "minioaccesskey"
minio_secret_key = "miniosecretkey"
minio_endpoint = "http://minio:9000"

spark = SparkSession.builder.appName("TrainALS").getOrCreate()

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

ratings = spark.read.csv(
    "s3a://movielens/train/ratings_train.csv",
    header=True,
    inferSchema=True,
)

train, val = ratings.randomSplit([0.8, 0.2])

als = ALS(
    maxIter=5,
    regParam=0.01,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
)
model = als.fit(train)

predictions = model.transform(val)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction",
)
rmse = evaluator.evaluate(predictions)
print(f"Validation RMSE: {rmse}")

model.write().overwrite().save("s3a://movielens/models/als_model")

spark.stop()
