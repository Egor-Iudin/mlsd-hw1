from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("AirflowTest").getOrCreate()
data = [("John", 25), ("Jane", 30), ("Mike", 35)]
columns = ["Name", "Age"]
df = spark.createDataFrame(data, columns)
df.show()
spark.stop()
