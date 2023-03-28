import configparser
import boto3
import io
from utils import logger
from parameters import CONTEXT

class S3():

    def __init__(self):
        config_file = 'config.ini'
        with open(config_file) as f:
            config = configparser.ConfigParser()
            config.read_file(f)

        ACCESS_KEY = config.get(CONTEXT, 'ACCESS_KEY')
        SECRET_KEY = config.get(CONTEXT, 'SECRET_KEY')
        REGION_AWS = config.get(CONTEXT, 'REGION_AWS')

        session = boto3.Session(
                    aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY,
                    region_name=REGION_AWS
                )
        self.s3 = session.client('s3')

    def list_objects(self, bucket_name, prefix):
        try:
            return self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        except Exception as e:
            logger.error(f"Error listing objects in S3 bucket: {e}")

    def download_file(self, bucket_name, obj):
        try:
            byte_stream = io.BytesIO()
            self.s3.download_fileobj(bucket_name, obj, byte_stream)
            byte_stream.seek(0)
            return byte_stream
        except Exception as e:
            logger.error(f"Error downloading objects from S3 bucket: {e}")