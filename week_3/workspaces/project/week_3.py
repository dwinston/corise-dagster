from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    SkipReason,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
    String,
)
from workspaces.config import REDIS, S3
from workspaces.project.sensors import get_s3_keys
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"},
)
def get_s3_data(context: OpExecutionContext) -> List[Stock]:
    return [Stock.from_list(record) for record in context.resources.s3.get_data(context.op_config["s3_key"])]


@op(
    out={"aggregation": Out(dagster_type=Aggregation)},
)
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
def put_s3_data(context: OpExecutionContext, aggregation: Aggregation):
    context.resources.s3.put_data(key_name=str(aggregation.date), data=aggregation)


@graph
def machine_learning_graph():
    agg = process_data(get_s3_data())
    put_redis_data(agg)
    put_s3_data(agg)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}


docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}

ONE_THROUGH_TEN = [str(n) for n in range(1, 11)]


@static_partitioned_config(partition_keys=ONE_THROUGH_TEN)
def docker_config(partition_key: str):
    return {
        "resources": {
            "s3": {"config": S3},
            "redis": {"config": REDIS},
        },
        "ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_key}.csv"}}},
    }


machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()},
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    op_retry_policy=RetryPolicy(max_retries=10, delay=1),
    config=docker_config,
    resource_defs={"s3": s3_resource, "redis": redis_resource},
)


machine_learning_schedule_local = ScheduleDefinition(
    job=machine_learning_job_local,
    cron_schedule="*/15 * * * *",
)


@schedule(cron_schedule="0 * * * *", job=machine_learning_job_docker)
def machine_learning_schedule_docker():
    for n in ONE_THROUGH_TEN:
        yield RunRequest(run_key=n)


@sensor(job=machine_learning_job_docker)
def machine_learning_sensor_docker():
    new_keys = get_s3_keys(bucket="dagster", prefix="prefix", endpoint_url="http://localstack:4566")
    if not new_keys:
        yield SkipReason("No new s3 files found in bucket.")
        return
    for key in new_keys:
        yield RunRequest(
            run_key=key,
            run_config={
               "resources": {
                    "s3": {"config": S3},
                    "redis": {"config": REDIS},
                },
                "ops": {"get_s3_data": {"config": {"s3_key": key}}},
            }
        )
