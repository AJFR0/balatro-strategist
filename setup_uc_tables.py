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

tables = ["jokers", "hands", "planets", "tarots", "spectrals", "vouchers", "decks", "tags",
          "blinds", "enhancements", "editions", "seals", "stakes",
          "joker_notes", "joker_benchmarks", "synergy_edges"]
COMMENT = {
    "jokers": "All 150 jokers: wiki-verified effect/cost/rarity (balatrowiki.org), type, activation, "
              "unlock condition, copyable/perishable/eternal compatibility, hand-written archetype + playbook.",
    "joker_notes": "Community wiki prose per joker (sections Synergies, Anti-Synergies, Strategy, Notes). "
                   "Text CC BY-NC-SA 3.0 balatrowiki.org.",
    "blinds": "Small/Big/Boss/Showdown blinds: effect, minimum ante, score multiplier vs base, $ reward.",
    "tarots": "22 Tarot cards ($3): effect + unlock.", "planets": "12 Planet cards ($3): hand levelled, chips/mult per level.",
    "spectrals": "18 Spectral cards ($4): effect; shop=0 means pack-only.",
    "vouchers": "32 vouchers ($10): tier Base/Upgraded, effect, prerequisite voucher.",
    "tags": "24 skip tags: effect, minimum ante.", "decks": "15 decks: effect + unlock.",
    "enhancements": "Card enhancements.", "editions": "Card/joker editions.", "seals": "Card seals.",
    "stakes": "8 stakes: difficulty modifier.",
}
for t in tables:
    pdf = pd.read_csv(os.path.join(data_dir, f"{t}.csv"))
    for c in pdf.columns:                       # no mixed str/NaN columns for Spark
        if pdf[c].dtype == object:
            pdf[c] = pdf[c].fillna("").astype(str)
    sdf = spark.createDataFrame(pdf)
    fq = f"`{catalog}`.`{schema}`.`{t}`"
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq)
    if t in COMMENT:
        spark.sql(f"COMMENT ON TABLE {fq} IS '{COMMENT[t].replace(chr(39), chr(39)*2)}'")
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
