# Databricks notebook source
# MAGIC %md
# MAGIC # Balatro Strategist — load the codex into Unity Catalog
# MAGIC Loads the `data/*.csv` files into UC tables so you can point a
# MAGIC **Genie space** at them and ask things like
# MAGIC *"which uncommon jokers under $6 synergize with a flush build?"*
# MAGIC
# MAGIC Run this notebook from the same folder as the app (it reads `./data/`).

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "balatro")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")

# COMMAND ----------

import os
import pandas as pd

here = os.path.dirname(os.path.abspath(dbutils.notebook.entry_point.getDbutils()
                        .notebook().getContext().notebookPath().get())) \
    if False else "."  # notebook cwd is the notebook's folder on serverless
data_dir = os.path.join(here, "data")

tables = ["jokers", "hands", "planets", "tarots", "spectrals", "vouchers", "decks", "tags"]
for t in tables:
    pdf = pd.read_csv(os.path.join(data_dir, f"{t}.csv"))
    sdf = spark.createDataFrame(pdf)
    fq = f"`{catalog}`.`{schema}`.`{t}`"
    sdf.write.mode("overwrite").saveAsTable(fq)
    print(f"wrote {fq}: {sdf.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: make it conversational
# MAGIC 1. Open **Genie** in the sidebar → **New Genie space**
# MAGIC 2. Add the `jokers` and `hands` tables (add the rest if you like)
# MAGIC 3. Ask: *"top 10 xMult jokers by rarity"*, *"which jokers reference flushes?"*,
# MAGIC    *"cheapest scaling jokers"* — Genie writes the SQL.

# COMMAND ----------

display(spark.sql(f"SELECT rarity, category, count(*) n FROM `{catalog}`.`{schema}`.jokers GROUP BY 1,2 ORDER BY n DESC"))
