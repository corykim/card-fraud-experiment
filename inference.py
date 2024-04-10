from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from data import load_data
from features import Features


def main():
    # Create a SparkSession
    spark = (
        SparkSession.builder.appName("CC-Fraud-Inference")
        .master("spark://vm-ubuntu:7077")
        .config("spark.executor.memory", "2g")
        .config("spark.submit.pyFiles", "./features.py,./data.py")
        .getOrCreate()
    )

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

    [inference_data] = data.randomSplit([0.05], seed=426)
    inference_data.show()

    # Load the saved pipeline
    model_path = "./pipelines/cc_fraud_pipeline"
    loaded_pipeline = PipelineModel.load(model_path)

    # Make predictions on the inference data
    predictions = loaded_pipeline.transform(inference_data)

    # Show the predictions
    predictions.select("merchant", "category", "prediction").show()

    positives = predictions.filter(predictions.prediction > 0)
    print("Positives:")
    positives.select("merchant", "category", "prediction").show()

    # Stop the SparkSession
    spark.stop()


if __name__ == "__main__":
    main()
