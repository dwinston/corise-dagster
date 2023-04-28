from datetime import datetime
from typing import List

from dagster import (
    AssetSelection,
    Nothing,
    OpExecutionContext,
    ScheduleDefinition,
    String,
    asset,
    define_asset_job,
    load_assets_from_current_module,
)
from workspaces.types import Aggregation, Stock


@asset(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"},
)
def get_s3_data(context: OpExecutionContext) -> List[Stock]:
    return [Stock.from_list(record) for record in context.resources.s3.get_data(context.op_config["s3_key"])]



@asset
def process_data(get_s3_data: List[Stock]) -> Aggregation:
    i_max, high_max = 0, -1
    for i, stock in enumerate(get_s3_data):
        if stock.high > high_max:
            i_max = i
            high_max = stock.high
    return Aggregation(date=get_s3_data[i_max].date, high=get_s3_data[i_max].high)



@asset(
    required_resource_keys={"redis"},
)
def put_redis_data(context: OpExecutionContext, process_data: Aggregation):
    context.resources.redis.put_data(name=str(process_data.date), value=str(process_data.high))

@asset(
    required_resource_keys={"s3"},
)
def put_s3_data(context: OpExecutionContext, process_data: Aggregation):
    context.resources.s3.put_data(key_name=str(process_data.date), data=process_data)


project_assets = load_assets_from_current_module()

local = {
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}

machine_learning_asset_job = define_asset_job(
    name="machine_learning_asset_job",
    selection=project_assets,
    config=local,
)

machine_learning_schedule = ScheduleDefinition(job=machine_learning_asset_job, cron_schedule="*/15 * * * *")
