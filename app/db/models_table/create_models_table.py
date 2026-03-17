from db_engine import engine
from sqlalchemy import Table, Column, Integer, String, MetaData, DateTime
from sqlalchemy.dialects.postgresql import JSON

metadata = MetaData()

models_table = Table(
    "models",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("fileName", String(50)),
    Column("type", String(50)),
    Column("dateTime", DateTime),
    Column("metrics", JSON)
)

metadata.create_all(engine)
