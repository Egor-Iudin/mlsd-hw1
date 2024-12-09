import shutil
import zipfile
from datetime import datetime
from json import loads
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.hooks.base_hook import BaseHook
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.log.logging_mixin import LoggingMixin

MOVIELENS_NAME = "ml-latest"
MOVIELENS_URL = f"https://files.grouplens.org/datasets/movielens/{MOVIELENS_NAME}.zip"

BUCKET_NAME = "movielens"

LOCAL_DATA_DIR = Path("/tmp/hw1") / BUCKET_NAME
TRAIN_DIR = LOCAL_DATA_DIR / "train"
TEST_DIR = LOCAL_DATA_DIR / "test"
TRAIN_RATIO = 0.8


logger = LoggingMixin().log

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


def download_and_unzip() -> None:
    if LOCAL_DATA_DIR.exists():
        shutil.rmtree(LOCAL_DATA_DIR)
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Movielens dataset from {MOVIELENS_URL}")

    r = requests.get(MOVIELENS_URL)
    r.raise_for_status()

    zip_path = LOCAL_DATA_DIR / f"{MOVIELENS_NAME}.zip"
    zip_path.write_bytes(r.content)

    logger.info("Download completed.")

    logger.info("Unzipping dataset...")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(LOCAL_DATA_DIR)

    logger.info("Unzipping completed.")


download_dataset_task = PythonOperator(
    task_id="download_and_unzip",
    python_callable=download_and_unzip,
    dag=dag,
)


def split_dataset() -> None:
    ratings_path = LOCAL_DATA_DIR / MOVIELENS_NAME / "ratings.csv"

    logger.info(f"Reading dataset from {ratings_path}")

    df = pd.read_csv(ratings_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp")

    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    train_file = TRAIN_DIR / "ratings_train.csv"
    test_file = TEST_DIR / "ratings_test.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(f"Train dataset saved to {train_file}")
    logger.info(f"Test dataset saved to {test_file}")


split_dataset_task = PythonOperator(
    task_id="split_dataset",
    python_callable=split_dataset,
    dag=dag,
)


def upload_dataset(
    local_dir: Path,
    s3_prefix: str,
    bucket_name: str,
    hook: S3Hook,
) -> None:
    for file in local_dir.glob("*"):
        if not file.is_file():
            continue

        key = f"{s3_prefix}/{file.name}"
        logger.info(f"Uploading {file} to s3://{bucket_name}/{key}")
        hook.load_file(
            filename=str(file),
            key=key,
            bucket_name=bucket_name,
            replace=True,
        )


def upload_datasets() -> None:
    hook = S3Hook(aws_conn_id="minios3")
    if not hook.check_for_bucket(BUCKET_NAME):
        logger.info(f"Bucket {BUCKET_NAME} does not exist. Creating...")
        hook.create_bucket(bucket_name=BUCKET_NAME)

    upload_dataset(TRAIN_DIR, "train", BUCKET_NAME, hook)
    upload_dataset(TEST_DIR, "test", BUCKET_NAME, hook)


upload_datasets_task = PythonOperator(
    task_id="upload_to_minio",
    python_callable=upload_datasets,
    dag=dag,
)

minios3_hook = BaseHook.get_connection("minios3")
minios3_env_vars = {
    "MINIO_ACCESS_KEY": minios3_hook.login,
    "MINIO_SECRET_KEY": minios3_hook.password,
    "MINIO_ENDPOINT": loads(minios3_hook.extra).get("endpoint_url"),
}
del minios3_hook

train_model_task = SparkSubmitOperator(
    task_id="train_model",
    application="/opt/airflow/scripts/train.py",
    conn_id="spark",
    jars="/opt/aws-java-sdk-bundle-1.12.540.jar,/opt/hadoop-aws-3.3.4.jar",
    dag=dag,
    env_vars=minios3_env_vars,
)

predict_model_task = SparkSubmitOperator(
    task_id="predict_model",
    application="/opt/airflow/scripts/predict.py",
    conn_id="spark",
    jars="/opt/aws-java-sdk-bundle-1.12.540.jar,/opt/hadoop-aws-3.3.4.jar",
    dag=dag,
    env_vars=minios3_env_vars,
)


def check_results() -> None:
    hook = S3Hook(aws_conn_id="minios3")

    logger.info("Listing prediction files in S3...")
    objects = hook.list_keys(
        bucket_name=BUCKET_NAME,
        prefix="predictions/raw_predictions",
    )
    if not objects:
        raise ValueError("No objects found in predictions directory.")

    part_files = [o for o in objects if "part-" in o]
    if not part_files:
        raise ValueError("No part file found in the predictions directory")

    part_key = part_files[0]
    logger.info(f"Downloading predictions from s3://{BUCKET_NAME}/{part_key}")
    local_pred_path = hook.download_file(key=part_key, bucket_name=BUCKET_NAME)

    df = pd.read_csv(local_pred_path)
    if "prediction_adj" not in df.columns:
        raise ValueError("Column 'prediction_adj' not found in predictions file.")

    logger.info("Predictions head:")
    logger.info(df.head().to_string())
    logger.info(f"Mean predicted rating: {df['prediction_adj'].mean()}")


check_results_task = PythonOperator(
    task_id="check_results",
    python_callable=check_results,
    dag=dag,
)


(
    download_dataset_task
    >> split_dataset_task
    >> upload_datasets_task
    >> train_model_task
    >> predict_model_task
    >> check_results_task
)
