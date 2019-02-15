import sys
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorIndexer, VectorAssembler, StringIndexer, OneHotEncoderEstimator
from pyspark.ml.evaluation import RegressionEvaluator

train_path = sys.argv[1]
train_labels_path = sys.argv[2]
#output_path = sys.argv[3]
#outfilename = output_path + "output.txt"
#output_file = open(outfilename, "a")

sc = pyspark.SparkContext(appName="dengAI")
spark = SparkSession.builder.getOrCreate()

data = pd.read_csv(train_path)
train_labels = pd.read_csv(train_labels_path)
data_trimmed = data.drop("precipitation_amt_mm", axis=1)
merged = pd.merge(data_trimmed, train_labels, how='inner')

# fill null values
merged['ndvi_ne'] = merged['ndvi_ne'].fillna((merged['ndvi_ne'].mean()))
merged['ndvi_nw'] = merged['ndvi_nw'].fillna((merged['ndvi_nw'].mean()))
merged['ndvi_se'] = merged['ndvi_se'].fillna((merged['ndvi_se'].mean()))
merged['ndvi_sw'] = merged['ndvi_sw'].fillna((merged['ndvi_sw'].mean()))
merged['reanalysis_air_temp_k'] = merged['reanalysis_air_temp_k'].fillna((merged['reanalysis_air_temp_k'].mean()))
merged['reanalysis_avg_temp_k'] = merged['reanalysis_avg_temp_k'].fillna((merged['reanalysis_avg_temp_k'].mean()))
merged['reanalysis_dew_point_temp_k'] = merged['reanalysis_dew_point_temp_k'].fillna((merged['reanalysis_dew_point_temp_k'].mean()))
merged['reanalysis_max_air_temp_k'] = merged['reanalysis_max_air_temp_k'].fillna((merged['reanalysis_max_air_temp_k'].mean()))
merged['reanalysis_min_air_temp_k'] = merged['reanalysis_min_air_temp_k'].fillna((merged['reanalysis_min_air_temp_k'].mean()))
merged['reanalysis_precip_amt_kg_per_m2'] = merged['reanalysis_precip_amt_kg_per_m2'].fillna((merged['reanalysis_precip_amt_kg_per_m2'].mean()))
merged['reanalysis_relative_humidity_percent'] = merged['reanalysis_relative_humidity_percent'].fillna((merged['reanalysis_relative_humidity_percent'].mean()))
merged['reanalysis_sat_precip_amt_mm'] = merged['reanalysis_sat_precip_amt_mm'].fillna((merged['reanalysis_sat_precip_amt_mm'].mean()))
merged['reanalysis_specific_humidity_g_per_kg'] = merged['reanalysis_specific_humidity_g_per_kg'].fillna((merged['reanalysis_specific_humidity_g_per_kg'].mean()))
merged['reanalysis_tdtr_k'] = merged['reanalysis_tdtr_k'].fillna((merged['reanalysis_tdtr_k'].mean()))
merged['station_avg_temp_c'] = merged['station_avg_temp_c'].fillna((merged['station_avg_temp_c'].mean()))
merged['station_diur_temp_rng_c'] = merged['station_diur_temp_rng_c'].fillna((merged['station_diur_temp_rng_c'].mean()))
merged['station_max_temp_c'] = merged['station_max_temp_c'].fillna((merged['station_max_temp_c'].mean()))
merged['station_min_temp_c'] = merged['station_min_temp_c'].fillna((merged['station_min_temp_c'].mean()))
merged['station_precip_mm'] = merged['station_precip_mm'].fillna((merged['station_precip_mm'].mean()))

spark_df = spark.createDataFrame(merged)
cols = spark_df.columns

categoricalColumns = ['city', 'week_start_date']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'total_cases', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['year', 'weekofyear', 'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'reanalysis_air_temp_k',
              'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
              'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
              'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c',
              'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(spark_df)
df = pipelineModel.transform(spark_df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
#df.printSchema()


train, test = df.randomSplit([0.75, 0.25], seed = 2018)
#print("Training Dataset Count: " + str(train.count()))
#print("Test Dataset Count: " + str(test.count()))

rf = RandomForestRegressor(featuresCol="features", labelCol='label')
rf_model = rf.fit(train)
predictions = rf_model.transform(test)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)
#print("Mean Absolute Error on test data for Random Forest Regressor = %g" % mae)
#output_file.write("Mean Absolute Error on test data for Random Forest Regressor = %g" % mae)
output = "Mean Absolute Error on test data for Random Forest Regressor = " + str(mae) + "\n"


gbt = GBTRegressor(maxIter=10, featuresCol="features", labelCol='label')
gb_model = gbt.fit(train)
preds = gb_model.transform(test)

mae = evaluator.evaluate(preds)
#print("Mean Absolute Error on test data for Gradient Boosted Regressor = %g" % mae)
#output_file.write("\nMean Absolute Error on test data for Gradient Boosted Regressor = %g" % mae)
output += "Mean Absolute Error on test data for Gradient Boosted Regressor =" + str(mae)
#rdd = sc.parallelize(output)
#rdd.coalesce(1,True).saveAsTextFile(output_path)
#sc.stop()











