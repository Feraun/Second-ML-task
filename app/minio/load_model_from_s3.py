import boto3
import joblib
import tempfile

s3 = boto3.client("s3")

BUCKET = "test-bucket"

