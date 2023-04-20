import boto3
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

class Adapter_Layer:
    def access(self):
        s3 = boto3.resource('s3')
        bucket = s3.Bucket('xetra-1234')
        arg_date = '2022-12-25'
        arg_date_dt = datetime.strptime(arg_date, '%Y-%m-%d').date() - timedelta(days=1)
        return bucket, arg_date_dt, s3

    def objects(self, bucket, arg_date_dt):
        return [obj for obj in bucket.objects.all() if
                datetime.strptime(obj.key.split('/')[0], '%Y-%m-%d').date() >= arg_date_dt]

    def first_df(self, bucket, objects, i):
        csv_obj_init = bucket.Object(key=objects[i].key).get().get('Body').read().decode('utf-8')
        data = StringIO(csv_obj_init)
        df_init = pd.read_csv(data, delimiter=",")
        return df_init