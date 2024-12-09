import os
import shutil
import zipfile
from datetime import datetime

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator


MOVIELENS_NAME = "ml-latest-small"
MOVIELENS_URL = f"https://files.grouplens.org/datasets/movielens/{MOVIELENS_NAME}.zip"

BUCKET_NAME = "movielens"

LOCAL_DATA_DIR = f"/tmp/hw1/{BUCKET_NAME}"
TRAIN_DIR = f"{LOCAL_DATA_DIR}/train"
TEST_DIR = f"{LOCAL_DATA_DIR}/test"


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "catchup": False,
}

dag = DAG(
    "movielens_pipeline",
    default_args=default_args,
    schedule_interval=None,
)


def download_and_unzip():
    if os.path.exists(LOCAL_DATA_DIR):
        shutil.rmtree(LOCAL_DATA_DIR)

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    zip_path = os.path.join(LOCAL_DATA_DIR, f"{MOVIELENS_NAME}.zip")

    r = requests.get(MOVIELENS_URL)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(LOCAL_DATA_DIR)


download_task = PythonOperator(
    task_id="download_and_unzip",
    python_callable=download_and_unzip,
    dag=dag,
)


def split_dataset():
    ratings_path = os.path.join(LOCAL_DATA_DIR, f"{MOVIELENS_NAME}/ratings.csv")
    df = pd.read_csv(ratings_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(TRAIN_DIR, "ratings_train.csv"), index=False)
    test_df.to_csv(os.path.join(TEST_DIR, "ratings_test.csv"), index=False)


split_task = PythonOperator(
    task_id="split_dataset",
    python_callable=split_dataset,
    dag=dag,
)


def upload_datasets():
    hook = S3Hook(aws_conn_id="minios3")
    if not hook.check_for_bucket(BUCKET_NAME):
        hook.create_bucket(bucket_name=BUCKET_NAME)

    for root, dirs, files in os.walk(TRAIN_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            key = f"train/{file}"
            hook.load_file(
                filename=local_path,
                key=key,
                bucket_name=BUCKET_NAME,
                replace=True,
            )

    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            key = f"test/{file}"
            hook.load_file(
                filename=local_path,
                key=key,
                bucket_name=BUCKET_NAME,
                replace=True,
            )


upload_task = PythonOperator(
    task_id="upload_to_minio",
    python_callable=upload_datasets,
    dag=dag,
)


train_model = SparkSubmitOperator(
    task_id="train_model",
    application="/opt/airflow/scripts/train.py",
    conn_id="spark",
    jars="/opt/aws-java-sdk-bundle-1.12.540.jar,/opt/hadoop-aws-3.3.4.jar",
    dag=dag,
)

predict_model = SparkSubmitOperator(
    task_id="predict_model",
    application="/opt/airflow/scripts/predict.py",
    conn_id="spark",
    jars="/opt/aws-java-sdk-bundle-1.12.540.jar,/opt/hadoop-aws-3.3.4.jar",
    dag=dag,
)


def check_results():
    hook = S3Hook(aws_conn_id="minios3")

    objects = hook.list_keys(
        bucket_name=BUCKET_NAME, prefix="predictions/raw_predictions"
    )
    part_file = [o for o in objects if "part-" in o]
    if not part_file:
        raise ValueError("No part file found in the predictions directory")

    part_key = part_file[0]

    pred = hook.download_file(
        key=part_key,
        bucket_name=BUCKET_NAME,
    )
    df = pd.read_csv(pred)
    print("Predictions head:")
    print(df.head())
    print("Mean predicted rating:", df["prediction_adj"].mean())


check_results_task = PythonOperator(
    task_id="check_results",
    python_callable=check_results,
    dag=dag,
)


(
    download_task
    >> split_task
    >> upload_task
    >> train_model
    >> predict_model
    >> check_results_task
)
