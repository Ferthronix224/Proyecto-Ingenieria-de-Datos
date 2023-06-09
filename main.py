import boto3  # Crear, configurar, y administrar servicios de AWS
import pandas as pd  # Manipulación y el análisis de datos
from io import StringIO, BytesIO  # StringIO - Este objeto se puede usar como entrada o salida para la mayoría de las
# funciones que esperarían un objeto de archivo estándar
# BytesIO - los datos se pueden guardar como bytes en un búfer en memoria cuando usamos las operaciones Byte IO del
# módulo io
from datetime import datetime, timedelta  # datetime - proporciona clases para manipular fechas y horas
# timedelta - representa una duración, la diferencia entre dos fechas u horas
import numpy as np  # crear vectores y matrices grandes multidimensionales, junto con una gran colección de funciones
# matemáticas de alto nivel para operar con ellas
from sklearn import linear_model  # Regresion lineal
from sklearn.model_selection import train_test_split  # Dividir arreglos o matrices en subconjuntos aleatorios de tren
# y prueba.
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import time  # proporciona varias funciones relacionadas con el tiempo


class Adapter_Layer:
    def access(self, s3, bucket, date):  # Método para acceder al servicio y al bucket
        s3 = boto3.resource(s3)  # Variable que inicializa el servicio de AWS
        bucket = s3.Bucket(bucket)  # Variable que inicializa el bucket del servicio
        arg_date_dt = datetime.strptime(date, '%Y-%m-%d').date() - timedelta(days=1)  # Variable que establece un
        # formato de fecha y se le recorre un dia
        return bucket, arg_date_dt, s3

    def objects(self, bucket, arg_date_dt):  # Método que devuelve los objetos de un bucket dentro de una fecha en
        # adelante
        return [obj for obj in bucket.objects.all() if datetime.strptime(obj.key.split('/')[0], '%Y-%m-%d').date() >=
                arg_date_dt]

    def first_df(self, bucket, objects):  # Método para sacar el primer dataframe de un objeto
        csv_obj_init = bucket.Object(key=objects[0].key).get().get('Body').read().decode('utf-8')  # Variable que
        # muestra la información del bucket
        data = StringIO(csv_obj_init)  # Variable que convierte la información del bucket a formato string
        df_init = pd.read_csv(data, delimiter=",")  # Creación del dataframe
        return df_init


class Application_Layer(Adapter_Layer):

    def first_df(self, bucket, objects):
        csv_obj_init = bucket.Object(key=objects[1].key).get().get('Body').read().decode('utf-8')
        data = StringIO(csv_obj_init)
        df_init = pd.read_csv(data, delimiter=",")
        return df_init

    def extract(self, df_init, objects, bucket):  # Método para hacer un dataframe de todos los buckets correspondientes
        df_all = pd.DataFrame(columns=df_init.columns)  # Variable que inicializa un dataframe con sus columnas
        for obj in objects:  # Ciclo que itera todos los objetos
            csv_obj = bucket.Object(key=obj.key).get().get('Body').read().decode('utf-8')  # Variable que muestra la
            # información del objeto
            data = StringIO(csv_obj)  # La información del objeto en formato string
            df = pd.read_csv(data, delimiter=',')  # Creación de dataframe del objeto iterado
            df_all = pd.concat([df, df_all], ignore_index=True)  # Concatenación del dataframe iterado en la variable
            # inicializada
        columns = ['ISIN', 'Mnemonic', 'Date', 'Time', 'StartPrice', 'EndPrice', 'MinPrice', 'MaxPrice', 'TradedVolume']
        # Nombres de las columnas que se quiere filtrar
        df_all = df_all.loc[:, columns]  # Filtración de las columnas

        return df_all

    def transform_report(self, df_all):
        df_all = df_all.loc[(df_all["Time"] >= '08:00') & (df_all["Time"] <= '12:00'), ["ISIN", "Date", "Time",
                                                                                        "StartPrice", "EndPrice"]]
        # Se filtran los registros entre las 8 y las 12 y se muestran las columnas establecidas
        std = []  # Variable para almacenar la desviación estandar
        List = [[df_all["StartPrice"]], [df_all["EndPrice"]]]  # Lista con las columnas necesarias para la desviación
        # estandar
        for i in range(len(List[0][0])):  # Ciclo for que itera el número de objetos
            Std = [List[0][0].iloc[i], List[1][0].iloc[i]]  # Iteración de las filas
            std.append(np.std(Std))  # agregar la desviación estandar con la variable creada anteriormente
        df_all['std'] = std  # Agregación de la desviación estándar
        df_all["EndPrice_MXN"] = df_all["EndPrice"] * 19.09  # EndPrice en pesos mexicanos
        df_all = df_all.sort_values(by=['Time'])  # Ordenar las filas en base a la hora
        df_all.round(decimals=4)  # Reducir los valores numericos a 4 decimales
        return df_all

    def load(self, df_all, s3):  # Método para subir a la nube el dataframe
        key = 'xetra_daily_report_' + datetime.today().strftime("%Y%m%d_%H%M%S") + '.parquet'  # Se establece el nombre
        # del archivo que se subirá a la nube
        out_buffer = BytesIO()  # Se inicializa variable para guardar información en el buffer
        df_all.to_parquet(out_buffer, index=False)  # Se convierte el dataframe a formato parquet
        bucket_target = s3.Bucket('ferthronix-xetra')  # Se inicializa el bucket a donde será almacenado el parquet
        bucket_target.put_object(Body=out_buffer.getvalue(), Key=key)  # Se sube el archivo a la nube

    def etl_report(self, s3):
        bucket_target = s3.Bucket('ferthronix-xetra')  # Se inicializa el bucket donde está almacenado el parquet
        parq = [obj.key for obj in bucket_target.objects.all()]  # Variable con los nombres de todos archivos parquet
        # en el bucket
        prq_obj = bucket_target.Object(key=parq[-1]).get().get('Body').read()  # Mostrar la información del último
        # parquet subido
        data = BytesIO(prq_obj)  # Variable con la información del parquet en el buffer
        df_report = pd.read_parquet(data)  # Variable que lee el archivo parquet
        return df_report

def neural_network(df_all):
    # codificar la columna de fechas
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df_all.sort_values('Date', inplace=True)
    df_all['Objetive'] = df_all['Date'].shift(-7)

    # dividir los datos en conjuntos de entrenamiento y prueba
    XX = np.array(df_all["Objetive"])
    X = []
    for i in XX:
        i = [i]
        X.append(i)
    y = df_all['EndPrice']  # objetivo

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Escalar las características utilizando un objeto de escalado de scikit-learn
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(1))
    # se crea una red neuronal secuencial con dos capas densas
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Se utiliza la función de pérdida de error cuadrático medio (MSE) y el optimizador Adam.

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)) # Entrenamiento

    predictions = model.predict(X_test) # Predicciones
    mse = mean_squared_error(y_test, predictions)  # Error cuadratico medio
    r2 = r2_score(y_test, predictions)  # Coeficiente de determinacion

    return predictions, mse, r2

def main():
    # Instancias
    adapter = Adapter_Layer()
    application = Application_Layer()
    # Parametros
    s3 = 's3'
    bucket = 'xetra-1234'
    date = '2022-12-25'
    # Adapter Layer
    bucket, arg_date_dt, s3 = adapter.access(s3, bucket, date)
    objects = adapter.objects(bucket, arg_date_dt)
    df = adapter.first_df(bucket, objects)
    # Application Layer
    df_all = application.extract(df, objects, bucket)
    transform = application.transform_report(df_all)
    load = application.load(transform, s3)
    etl_report = application.etl_report(s3)

    print(etl_report)
    neural = neural_network(transform)
    print(neural[0])
    print(neural[1])
    print(neural[2])

inicio = time.time()
main()
fin = time.time()
print(int(fin - inicio))