import boto3
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score