from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime

# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

# Define the DAG
with DAG(
    "test_pyspark",
    default_args=default_args,
    description="A simple test for PySpark with Airflow",
    schedule_interval=None,  # Run on demand
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["example", "pyspark"],
) as dag:

    # SparkSubmitOperator to submit the PySpark job
    test_pyspark_job = SparkSubmitOperator(
        task_id="run_pyspark_job",
        application="/opt/airflow/scripts/test_spark.py",  # Path to your PySpark script
        conn_id="spark",  # Connection ID configured in Airflow
        verbose=True,
        application_args=["arg1", "arg2"],  # Optional: Arguments to pass to the script
        executor_cores=2,
        total_executor_cores=4,
        name="test_pyspark_job",
        driver_memory="2g",
        executor_memory="2g",
    )

    test_pyspark_job
