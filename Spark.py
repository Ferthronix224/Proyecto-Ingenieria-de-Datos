from pyspark.sql import SparkSession
from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import mean
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.getOrCreate()
url = 'https://raw.githubusercontent.com/Ferthronix224/Deaths-and-Births/main/births-and-deaths-projected-to-2100.csv'
df = pd.read_csv(url)
df_pyspark = spark.createDataFrame(df)
df_pyspark.show()

print('Filtrar resultados que sean mayores al 2023')
df_pyspark.filter("Year>2023").show()

print('Mostrar las columnas de Entity y Deaths_medium del año 2023')
df_pyspark.filter("Year==2023").select(["Entity", "Deaths_medium"]).show()

print('Filtrar los registros de los últimos 10 años en México')
df_pyspark.filter((df_pyspark["Year"]>=2013) & (df_pyspark["Year"]<=2023) & (df_pyspark["Entity"]=='Mexico')).show()

print('Filtrar Mexico, Canada y Estados Unidos y agrupar sus valores maximos')
df_pyspark.filter((df_pyspark["Entity"]=='United States') | (df_pyspark["Entity"]=='Canada') | (df_pyspark["Entity"]=='Mexico')).groupBy("Entity").max().show()

print('GroupBy (Sum)')
df_without_year = spark.createDataFrame(df, ['Entity', 'Deaths_estimates', 'Deaths_medium', 'Births_estimates', 'Births_medium'])
df_without_year.groupBy("Entity").sum().show()

print('GroupBy (Count)')
df_pyspark.groupBy("Entity").count().show()

print('GroupBy (Mean)')
df_pyspark.groupBy("Deaths_estimates").mean().show()

print('GroupBy (Max)')
df_without_year.groupBy("Entity").max().show()

print('Monotonically Increasing Id')
df = df_pyspark.withColumn('id', monotonically_increasing_id())
df = df[['id'] +  df.columns[:-1]]
df.show(3)

print('Conteo')
print(df.count())

print('Promedio de Nacimientos en general')
df.select('Births_estimates').agg({'Births_estimates': 'avg'}).show()

print('Promedio de diferentes valores')
df.select(*[mean(c) for c in df.columns]).show()

print('Se hace un train de 70 porciento de los valores y un test del 30 porciento de los valores')
train, test = df.randomSplit([0.7, 0.3])
print(train, test)

numerical_features_list = train.columns
numerical_features_list.remove('Entity')
numerical_features_list.remove('Code')
print('Se borran columnas no numericas')
numerical_features_list

print('Uso del Imputer')
imputer = Imputer(inputCols=numerical_features_list, outputCols=numerical_features_list)

imputer = imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

train.show()

print('Uso de VectorAssembler')
numerical_vector_assembler = VectorAssembler(inputCols=numerical_features_list, outputCol='numerical_feature_vector')

train = numerical_vector_assembler.transform(train)
test = numerical_vector_assembler.transform(test)
print(train.select('numerical_feature_vector').take(2))
print()

print('Uso de StandardScaler')
scaler = StandardScaler(inputCol='numerical_feature_vector',
                      outputCol='scaled_numerical_feature_vector',
                      withStd=True, withMean=True)
scaler = scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)
print(train.select('scaled_numerical_feature_vector').take(3))
print()

print('Uso de StringIndexer')

indexer = StringIndexer(inputCol='Entity',
                      outputCol='Entity_Index')
indexer = indexer.fit(train)
train = indexer.transform(train)
test = indexer.transform(test)
print(train.select('Entity_Index').take(2))
print()

print('Uso de One Hot Encoder')
ohe = OneHotEncoder(inputCol='Entity_Index', outputCol='entity_ohe')

ohe = ohe.fit(train)
train = ohe.transform(train)
test = ohe.transform(test)
print(train.select('entity_ohe').take(2))
print()

print('Assembler')
assembler = VectorAssembler(inputCols=['scaled_numerical_feature_vector', 'entity_ohe'], outputCol='final_feature_col')
train = assembler.transform(train)
test = assembler.transform(test)
print(train.select('final_feature_col').take(2))
print()

print('Regresion Linear')
lr = LinearRegression(featuresCol='final_feature_col', labelCol='Deaths_medium')
lr = lr.fit(train)

print('Datos de entrenamiento')
pred_train_df = lr.transform(train).withColumnRenamed('prediction', 'predicted_value')
print(pred_train_df.select('predicted_value').take(2))
print()

print('Datos de prueba')
pred_test_df = lr.transform(test).withColumnRenamed('prediction', 'predicted_value')
print(pred_test_df.select('predicted_value').take(2))
print()

print('Representación de valores predecidos y reales')
predictions_and_actuals = pred_test_df['predicted_value', 'Deaths_medium']
predictions_and_actuals_rdd = predictions_and_actuals.rdd
print(predictions_and_actuals_rdd.take(2))