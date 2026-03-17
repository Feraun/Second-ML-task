import tempfile

import boto3
import joblib
from botocore.client import Config

class S3BucketService:

    def __init__(self, bucket, endpoint, access, secret):
        self.bucket = bucket

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            config=Config(signature_version="s3v4"),
        )

    def upload_file(self, local_path: str, object_name: str):
        self.client.upload_file(
            local_path,
            self.bucket,
            object_name
        )

    def list_objects(self, prefix: str):
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )
        return [obj["Key"] for obj in response.get("Contents", [])]

    def delete_object(self, key: str):
        self.client.delete_object(
            Bucket=self.bucket,
            Key=key
        )

    def load_model_from_s3(self, s3_path: str):

        with tempfile.NamedTemporaryFile() as tmp:

            self.client.download_file(
                self.bucket,
                s3_path,
                tmp.name
            )

            model = joblib.load(tmp.name)

        return model