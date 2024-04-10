from math import radians, cos, sin, asin, sqrt

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import udf, lag, coalesce, lit
from pyspark.sql.functions import unix_timestamp, col, avg, count
from pyspark.sql.types import FloatType


# User-defined function to calculate the distance between two latitude-longitude pairs
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of the Earth in kilometers
    return c * r


# Register the haversine distance UDF
haversine_udf = udf(haversine_distance, returnType=FloatType())


# Syntactic sugar to make it easier to add features in a fluent coding style
class Features:
    data: DataFrame = None

    def __init__(self, data: DataFrame):
        self.data = data

    def get_data(self) -> DataFrame:
        return self.data

    # Calculate the distance between the user and merchant locations
    def with_distance(self):
        self.data = self.data.withColumn(
            "distance", haversine_udf("lat", "long", "merch_lat", "merch_long")
        )
        return self

    # convert the transaction timestamp to a Unix timestamp
    def with_timestamp(self):
        # Convert the transaction timestamp to Unix timestamp
        self.data = self.data.withColumn(
            "trans_timestamp",
            unix_timestamp(col("trans_date_trans_time"), "dd/MM/yyyy HH:mm"),
        )
        return self


    # Calculate the time since the last transaction. Requires with_timestamp()
    def with_time_since_last_transaction(self):
        # Calculate the time since the last transaction for each credit card
        window_spec = Window.partitionBy("cc_num").orderBy("trans_timestamp")
        self.data = self.data.withColumn(
            "time_since_last_trans",
            coalesce(
                col("trans_timestamp") - lag("trans_timestamp", 1).over(window_spec),
                lit(0),
            ),
        )
        return self

    # Calculate transaction statistics. Requires with_timestamp()
    def with_transaction_statistics(self):
        data = self.data

        # Calculate the average transaction amount and frequency over a certain period (e.g., last 30 days)
        window_spec = (
            Window.partitionBy("cc_num")
            .orderBy("trans_timestamp")
            .rangeBetween(-30 * 24 * 60 * 60, 0)
        )
        data = data.withColumn("avg_amount_last_30d", avg("amt").over(window_spec))
        data = data.withColumn("trans_freq_last_30d", count("*").over(window_spec))

        self.data = data
        return self
