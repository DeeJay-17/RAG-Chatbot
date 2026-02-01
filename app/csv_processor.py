# # app/csv_processor.py

import pandas as pd
import uuid
from sqlalchemy import Table, Column, MetaData
from sqlalchemy import String, Integer, Float

def infer_sqlalchemy_type(dtype):
    if "int" in str(dtype):
        return Integer
    elif "float" in str(dtype):
        return Float
    else:
        return String

def process_csv(file_path, original_filename, engine):
    doc_id = str(uuid.uuid4())
    dynamic_table_name = doc_id 

    df = pd.read_csv(file_path)
    metadata = MetaData()
    columns = []

    for col_name, dtype in df.dtypes.items():
        columns.append(
            Column(col_name, infer_sqlalchemy_type(dtype))
        )

    csv_table = Table(dynamic_table_name, metadata, *columns)

    metadata.create_all(engine)

    df.to_sql(dynamic_table_name, engine, if_exists="append", index=False)

    return doc_id, dynamic_table_name