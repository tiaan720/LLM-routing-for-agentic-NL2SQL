import argparse
import os
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime

from google.cloud import storage

BUCKET_NAME = "mlflow-llm-routing"


def upload_file(client, bucket, local_path, blob_name):
    """Upload a file to the bucket."""
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to {blob_name}")


def upload_folder(client, bucket, local_folder, blob_prefix):
    """Upload a folder as a zip file to the bucket."""
    zip_path = os.path.join(
        tempfile.gettempdir(), f"{os.path.basename(local_folder)}.zip"
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                arcname = os.path.relpath(local_path, local_folder)
                zipf.write(local_path, arcname)
    blob_name = f"{blob_prefix.rstrip('/')}.zip"
    upload_file(client, bucket, zip_path, blob_name)
    os.remove(zip_path)  # Clean up temp zip


def download_file(client, bucket, blob_name, local_path):
    """Download a file from the bucket."""
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded {blob_name} to {local_path}")


def download_folder(client, bucket, blob_prefix, local_folder):
    """Download a zip file from the bucket and unzip it."""
    blob_name = f"{blob_prefix.rstrip('/')}.zip"
    zip_path = os.path.join(
        tempfile.gettempdir(), f"{os.path.basename(local_folder)}.zip"
    )
    download_file(client, bucket, blob_name, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(local_folder)
    os.remove(zip_path)  # Clean up temp zip


def clean_artifact_paths(db_path="mlflow.db"):
    """Clean Windows artifact paths to Linux paths in the MLflow database."""
    if not os.path.exists(db_path):
        print(f"{db_path} not found, skipping path cleaning.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check for Windows paths
    cursor.execute("SELECT COUNT(*) FROM runs WHERE artifact_uri LIKE 'file:///C:%'")
    count = cursor.fetchone()[0]
    if count == 0:
        print("No Windows paths found, skipping.")
        conn.close()
        return

    print(f"Found {count} runs with Windows paths, updating...")

    # Get current directory for new prefix
    current_dir = os.getcwd()
    new_prefix = f"file://{current_dir}"

    # Update paths: replace everything before '/mlruns' with new_prefix
    cursor.execute(
        """
        UPDATE runs 
        SET artifact_uri = ? || SUBSTR(artifact_uri, INSTR(artifact_uri, '/mlruns'))
        WHERE artifact_uri LIKE 'file:///C:%'
    """,
        (new_prefix,),
    )
    conn.commit()

    updated_count = cursor.rowcount
    print(f"Updated {updated_count} artifact paths.")

    conn.close()


def get_latest_date(client, bucket):
    """Get the latest timestamp folder in the bucket."""
    blobs = bucket.list_blobs()
    timestamps = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) > 0:
            ts_str = parts[0]
            try:
                datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")
                timestamps.add(ts_str)
            except ValueError:
                pass
    if timestamps:
        # Sort by datetime and get the latest
        latest_ts = max(
            timestamps, key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S")
        )
        return latest_ts
    else:
        raise ValueError("No timestamped folders found in the bucket.")


def push(client, bucket, ts_str):
    """Push mlflow.db and mlruns folder to the bucket with timestamp prefix."""
    # Upload mlflow.db
    if os.path.exists("mlflow.db"):
        upload_file(client, bucket, "mlflow.db", f"{ts_str}/mlflow.db")
    else:
        print("mlflow.db not found, skipping.")

    # Upload mlruns folder
    if os.path.exists("mlruns"):
        upload_folder(client, bucket, "mlruns", f"{ts_str}/mlruns")
    else:
        print("mlruns folder not found, skipping.")


def pull(client, bucket, ts_str=None):
    """Pull mlflow.db and mlruns folder from the bucket."""
    if ts_str is None:
        ts_str = get_latest_date(client, bucket)
        print(f"Pulling latest: {ts_str}")

    # Download mlflow.db
    blob_name = f"{ts_str}/mlflow.db"
    if bucket.blob(blob_name).exists():
        download_file(client, bucket, blob_name, "mlflow.db")
    else:
        print(f"{blob_name} not found, skipping.")

    # Download mlruns folder
    blob_name = f"{ts_str}/mlruns.zip"
    if bucket.blob(blob_name).exists():
        # Remove existing mlruns if exists
        if os.path.exists("mlruns"):
            shutil.rmtree("mlruns")
        download_folder(client, bucket, f"{ts_str}/mlruns/", "mlruns")
    else:
        print(f"{blob_name} not found, skipping.")

    # Clean artifact paths
    clean_artifact_paths()


def main():
    parser = argparse.ArgumentParser(
        description="Push or pull MLflow data to/from GCP bucket."
    )
    parser.add_argument(
        "--push", action="store_true", help="Push MLflow data to the bucket"
    )
    parser.add_argument(
        "--pull", action="store_true", help="Pull MLflow data from the bucket"
    )
    parser.add_argument(
        "--date",
        help="Timestamp in YYYY-MM-DD_HH-MM-SS format for pull (optional, defaults to latest)",
    )

    args = parser.parse_args()

    if not (args.push or args.pull):
        parser.error("You must specify either --push or --pull")

    # Initialize client
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    if args.push:
        ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        push(client, bucket, ts_str)
    elif args.pull:
        ts_str = args.date
        pull(client, bucket, ts_str)


if __name__ == "__main__":
    main()
