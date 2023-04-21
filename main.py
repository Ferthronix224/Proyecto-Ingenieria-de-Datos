import boto3
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
import time


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


class Application_Layer():

    def extract(self, df_init, objects, bucket):
        df_all = pd.DataFrame(columns=df_init.columns)
        for obj in objects:
            csv_obj = bucket.Object(key=obj.key).get().get('Body').read().decode('utf-8')
            data = StringIO(csv_obj)
            df = pd.read_csv(data, delimiter=',')
            df_all = pd.concat([df, df_all], ignore_index=True)
        columns = ['ISIN', 'Mnemonic', 'Date', 'Time', 'StartPrice', 'EndPrice', 'MinPrice', 'MaxPrice', 'TradedVolume']
        df_all = df_all.loc[:, columns]

        return df_all

    def transform_report(self, df_all):
        df_all = df_all.loc[
            (df_all["Time"] >= '08:00') & (df_all["Time"] <= '12:00'), ["ISIN", "Date", "Time", "StartPrice",
                                                                        "EndPrice"]]
        std = []
        List = [[df_all["StartPrice"]], [df_all["EndPrice"]]]
        for i in range(len(List[0][0])):
            Std = [List[0][0].iloc[i], List[1][0].iloc[i]]
            std.append(np.std(Std))
        df_all['std'] = std
        df_all["EndPrice_MXN"] = df_all["EndPrice"] * 19.09
        df_all = df_all.sort_values(by=['Time'])
        df_all.round(decimals=4)
        return df_all

    def load(self, df_all, s3):
        key = 'xetra_daily_report_' + datetime.today().strftime("%Y%m%d_%H%M%S") + '.parquet'
        out_buffer = BytesIO()
        df_all.to_parquet(out_buffer, index=False)
        bucket_target = s3.Bucket('ferthronix-xetra')
        bucket_target.put_object(Body=out_buffer.getvalue(), Key=key)

    def etl_report(self, s3):
        bucket_target = s3.Bucket('ferthronix-xetra')
        parq = [obj.key for obj in bucket_target.objects.all()]
        prq_obj = bucket_target.Object(key=parq[-1]).get().get('Body').read()
        data = BytesIO(prq_obj)
        df_report = pd.read_parquet(data)
        return df_report

    def linear_regression(df_all):
        # Se asigna variable de entrada X para entrenamiento y las etiquetas Y.
        dataX = df_all['Time'].replace({':': '.'}, regex=True).astype(float)
        XX = np.array(dataX)
        X_train = []
        for i in XX:
            i = [i]
            X_train.append(i)
        y_train = df_all['EndPrice'].values

        # Creamos el objeto de Regresión Linear
        regr = linear_model.LinearRegression()

        # Entrenamos nuestro modelo
        regr.fit(X_train, y_train)

        # Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
        y_pred = regr.predict(X_train)

        # Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
        print('Coefficients: \n', regr.coef_)
        # Este es el valor donde corta el eje Y (en X=0)
        print('Independent term: \n', regr.intercept_)
        # Error Cuadrado Medio
        print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
        # Puntaje de Varianza. El mejor puntaje es un 1.0
        print('Variance score: %.2f' % r2_score(y_train, y_pred))

        prediccion = regr.predict([[24]])
        print(prediccion)

def neural_network(df_all):
    X = df_all.drop(['Date', 'ISIN', 'StartPrice', 'Time'], axis=1)  # características
    y = df_all['EndPrice']  # objetivo

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    predictions = model.predict(X_test)

    return predictions

def main():
    # Instancias
    adapter = Adapter_Layer()
    application = Application_Layer()
    # Adapter Layer
    bucket, arg_date_dt, s3 = adapter.access()
    objects = adapter.objects(bucket, arg_date_dt)
    df = adapter.first_df(bucket, objects, 10)
    # Application Layer
    df_all = application.extract(df, objects, bucket)
    transform = application.transform_report(df_all)
    load = application.load(transform, s3)
    etl_report = application.etl_report(s3)

    print(etl_report)
    print(neural_network(transform))

inicio = time.time()
main()
fin = time.time()
print(int(fin - inicio))