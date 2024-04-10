from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import SparkSession

from data import load_data
from features import Features


def main():
    # Create a SparkSession
    spark = (
        SparkSession.builder.appName("CC-Fraud-Test")
        .master("spark://vm-ubuntu:7077")
        .config("spark.executor.memory", "2g")
        .config("spark.submit.pyFiles", "./features.py,./data.py")
        .getOrCreate()
    )

    # load the raw data into a DataFrame
    data = load_data(spark)

    # add derived features
    data = (
        Features(data)
        .with_distance()
        .with_timestamp()
        .with_time_since_last_transaction()
        .with_transaction_statistics()
        .get_data()
    )

    print("Data after derived features:")
    data.show()

    # Select relevant features
    selected_features = [
        "amt",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
        "distance",
        "time_since_last_trans",
        "avg_amount_last_30d",
        "trans_freq_last_30d",
        "gender_index",
        "category_code",
        "state_index",
        "job_index",
        "merchant_index",
    ]

    # Create a StringIndexer for string values
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
    category_indexer = StringIndexer(inputCol="category", outputCol="category_index")
    state_indexer = StringIndexer(inputCol="state", outputCol="state_index")
    job_indexer = StringIndexer(inputCol="job", outputCol="job_index")
    merchant_indexer = StringIndexer(inputCol="merchant", outputCol="merchant_index")

    # Create a OneHotEncoder for the indexed "category" field
    category_encoder = OneHotEncoder(
        inputCol="category_index", outputCol="category_code"
    )

    # Create a VectorAssembler to combine selected features
    assembler = VectorAssembler(
        inputCols=selected_features,
        outputCol="features",
    )

    # Create a StandardScaler to scale the features
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withStd=True, withMean=False
    )

    # Create a LogisticRegression model
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="is_fraud")

    # Create a pipeline
    pipeline = Pipeline(
        stages=[
            gender_indexer,
            category_indexer,
            category_encoder,
            state_indexer,
            job_indexer,
            merchant_indexer,
            assembler,
            scaler,
            lr,
        ]
    )

    # Split the data into training and testing sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    # Fit the pipeline to the training data
    model = pipeline.fit(train_data)

    # Make predictions on the testing data
    predictions = model.transform(test_data)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(labelCol="is_fraud")
    auc = evaluator.evaluate(predictions)
    print("AUC:", auc)

    # Show some sample predictions
    print("Test data count: ", test_data.count())
    print(
        "Incorrect predictions: ",
        predictions.filter(predictions.is_fraud != predictions.prediction).count(),
    )

    correct = predictions.filter(predictions.is_fraud != 0).filter(
        predictions.is_fraud == predictions.prediction
    )
    print("\n\nTrue positives:", correct.count())
    print("===================")
    correct.select("prediction", "is_fraud").show()

    model_path = "./pipelines/cc_fraud_pipeline"
    print(f"Saving model to {model_path}...")
    model.write().overwrite().save(model_path)
    print("Model saved.")

    # Stop the SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
