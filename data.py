from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    IntegerType,
)

# https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction
DATA_PATH = "./data/card-fraud/fraud-test.csv"


def load_data(spark: SparkSession):
    print("Loading data ...")

    schema = StructType(
        [
            StructField("row_id", IntegerType(), False),
            StructField("trans_date_trans_time", StringType(), False),
            StructField("cc_num", StringType(), False),
            StructField("merchant", StringType(), False),
            StructField("category", StringType(), False),
            StructField("amt", FloatType(), False),
            StructField("first", StringType(), False),
            StructField("last", StringType(), False),
            StructField("gender", StringType(), False),
            StructField("street", StringType(), False),
            StructField("city", StringType(), False),
            StructField("state", StringType(), False),
            StructField("zip", StringType(), False),
            StructField("lat", FloatType(), False),
            StructField("long", FloatType(), False),
            StructField("city_pop", IntegerType(), False),
            StructField("job", StringType(), False),
            StructField("dob", StringType(), False),
            StructField("trans_num", StringType(), False),
            StructField("unix_time", IntegerType(), False),
            StructField("merch_lat", FloatType(), False),
            StructField("merch_long", FloatType(), False),
            StructField("is_fraud", IntegerType(), False),
        ]
    )

    df = spark.read.csv(DATA_PATH, header=True, schema=schema)
    return df
