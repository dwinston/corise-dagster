from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    String,
    graph,
    op,
)
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"},
)
def get_s3_data(context: OpExecutionContext) -> List[Stock]:
    return [Stock.from_list() for record in context.resources.s3.get_data(context.op_config["s3_key"])]


@op
def process_data(stocks: List[Stock]) -> Aggregation:
    i_max, high_max = 0, -1
    for i, stock in enumerate(stocks):
        if stock.high > high_max:
            i_max = i
            high_max = stock.high
    return Aggregation(date=stocks[i_max].date, high=stocks[i_max].high)


@op(
    required_resource_keys={"redis"},
)
def put_redis_data(context: OpExecutionContext, aggregation: Aggregation):
    context.resources.redis.put_data(name=str(aggregation.date), value=str(aggregation.high))


@op(
    required_resource_keys={"s3"},
)
def put_s3_data(ontext: OpExecutionContext, aggregation: Aggregation):
    context.resources.s3.put_data(key_name=str(aggregation.date), data=aggregation)


@graph
def machine_learning_graph():
    agg = process_data(get_s3_data())
    put_redis_data(agg)
    put_s3_data(agg)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()},
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker,
    resource_defs={"s3": s3_resource, "redis": redis_resource},
)
