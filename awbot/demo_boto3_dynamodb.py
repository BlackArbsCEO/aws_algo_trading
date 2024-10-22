import boto3

table_name = "price_history_table"

table = boto3.resource("dynamodb", region_name="us-east-1").Table(table_name)
print(table)
