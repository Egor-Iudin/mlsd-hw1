from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import requests
import zipfile
import os
import logging

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}

dag = DAG(
    "download_movielens",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="Download, unzip, and upload MovieLens dataset to Minio",
)


def download_dataset(**context):
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    local_zip_path = "/tmp/ml-latest-small.zip"
    logging.info(f"Downloading dataset from {url}")
    response = requests.get(url)
    response.raise_for_status()  # Raise an error on bad status
    with open(local_zip_path, "wb") as f:
        f.write(response.content)
    logging.info(f"Dataset downloaded to {local_zip_path}")
    context["ti"].xcom_push(key="local_zip_path", value=local_zip_path)


def unzip_dataset(**context):
    local_zip_path = context["ti"].xcom_pull(
        key="local_zip_path", task_ids="download_dataset"
    )
    extract_path = "/tmp/ml-latest-small"
    logging.info(f"Unzipping dataset {local_zip_path} to {extract_path}")
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    logging.info("Dataset unzipped successfully")
    context["ti"].xcom_push(key="extract_path", value=extract_path)


def upload_to_minio(**context):
    extract_path = context["ti"].xcom_pull(key="extract_path", task_ids="unzip_dataset")
    bucket_name = "movielens"
    s3_hook = S3Hook(aws_conn_id="minio_s3")

    # Create bucket if it doesn't exist
    if not s3_hook.check_for_bucket(bucket_name):
        logging.info(f"Bucket {bucket_name} does not exist. Creating bucket.")
        s3_hook.create_bucket(bucket_name)

    # Upload files to Minio
    logging.info(f"Uploading files from {extract_path} to Minio bucket {bucket_name}")
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.relpath(local_file_path, extract_path)
            s3_hook.load_file(
                filename=local_file_path,
                key=s3_key,
                bucket_name=bucket_name,
                replace=True,
            )
            logging.info(f"Uploaded {s3_key} to bucket {bucket_name}")
    logging.info("All files uploaded successfully")


download_task = PythonOperator(
    task_id="download_dataset",
    python_callable=download_dataset,
    provide_context=True,
    dag=dag,
)

unzip_task = PythonOperator(
    task_id="unzip_dataset",
    python_callable=unzip_dataset,
    provide_context=True,
    dag=dag,
)

upload_task = PythonOperator(
    task_id="upload_to_minio",
    python_callable=upload_to_minio,
    provide_context=True,
    dag=dag,
)

download_task >> unzip_task >> upload_task
