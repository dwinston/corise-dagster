import csv
from datetime import datetime
from typing import Iterator, List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    String,
    job,
    op,
    usable_as_dagster_type,
)
from pydantic import BaseModel


@usable_as_dagster_type(description="Stock data")
class Stock(BaseModel):
    date: datetime
    close: float
    volume: int
    open: float
    high: float
    low: float

    @classmethod
    def from_list(cls, input_list: List[str]):
        """Do not worry about this class method for now"""
        return cls(
            date=datetime.strptime(input_list[0], "%Y/%m/%d"),
            close=float(input_list[1]),
            volume=int(float(input_list[2])),
            open=float(input_list[3]),
            high=float(input_list[4]),
            low=float(input_list[5]),
        )


@usable_as_dagster_type(description="Aggregation of stock data")
class Aggregation(BaseModel):
    date: datetime
    high: float


def csv_helper(file_name: str) -> Iterator[Stock]:
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield Stock.from_list(row)


@op(config_schema={"s3_key": String})
def get_s3_data_op(context) -> List[Stock]:
    """Get a list of stocks from an S3 file."""
    return list(csv_helper(context.op_config["s3_key"]))


@op
def process_data_op(stocks: List[Stock]) -> Aggregation:
    i_max, high_max = 0, -1
    for i, stock in enumerate(stocks):
        if stock.high > high_max:
            i_max = i
            high_max = stock.high
    return Aggregation(date=stocks[i_max].date, high=stocks[i_max].high)


@op
def put_redis_data_op(aggregation: Aggregation):
    pass


@op
def put_s3_data_op(aggregation: Aggregation):
    pass


@job
def machine_learning_job():
    agg = process_data_op(get_s3_data_op())
    put_redis_data_op(agg)
    put_s3_data_op(agg)
