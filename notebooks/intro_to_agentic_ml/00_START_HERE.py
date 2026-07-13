# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="background:#1B3139; color:white; padding:24px; border-radius:8px;">
# MAGIC   <h1 style="margin:0; color:white;">▶ START HERE — ML on Databricks workshop</h1>
# MAGIC   <p style="margin:6px 0 0 0; color:#9EB7BE; font-size:16px;">
# MAGIC     Run this one notebook top to bottom, then open the labs. No CLI, no asset bundle — just notebooks.
# MAGIC   </p>
# MAGIC </div>
# MAGIC
# MAGIC This notebook does the three setup steps for you, in order:
# MAGIC 1. **Choose your catalog** and create the catalog / schemas / landing volume.
# MAGIC 2. **Generate the synthetic data** the labs read (sales + brewery history).
# MAGIC 3. Point you to the labs (and tell you when to run the streaming producer for Lab 2).
# MAGIC
# MAGIC Attach any **serverless** notebook compute (environment version 5). Everything here is CPU-only.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Your catalog & workspace
# MAGIC 👉 **Run the next cell.** A **`1. Catalog` widget then appears at the very top of this
# MAGIC notebook**, pre-filled with your own catalog **`workshop_firstname_lastname`**. That widget is
# MAGIC the *only* place you set your catalog — every lab reads it via `%run ./src/00_setup`.
# MAGIC
# MAGIC - **Keep the pre-filled value** → your own isolated sandbox (created for you).
# MAGIC - **Type a different name** → use a shared/existing catalog instead.
# MAGIC
# MAGIC If you change the widget, **re-run** so the new catalog takes effect. You need
# MAGIC `CREATE CATALOG` for a brand-new catalog, or have an admin create it and grant you access.

# COMMAND ----------

# DBTITLE 1,Set your catalog — this puts the "1. Catalog" widget at the TOP of this notebook
# Run this cell FIRST. It declares the `catalog` widget right here in START HERE (so you can set
# it before anything else runs), pre-filled from your login as workshop_<firstname>_<lastname>.
# The same-named widget is read by ./src/00_setup and by every lab, so whatever you set here
# flows everywhere. Change it in the widget bar at the top, then Run all.
from databricks.sdk.runtime import dbutils, spark
import re

_user = spark.sql("SELECT current_user()").collect()[0][0]
_slug = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", _user.split("@")[0].lower())).strip("_")
dbutils.widgets.text("catalog", f"workshop_{_slug}" if _slug else "ml_workshop", "1. Catalog (your sandbox)")
print(f"Catalog widget is at the top of the notebook. Current value: {dbutils.widgets.get('catalog')!r}")
print("→ Keep it, or type a different catalog in the widget, THEN Run all.")

# COMMAND ----------

# MAGIC %run ./src/00_setup

# COMMAND ----------

# DBTITLE 1,Confirm where everything will live
print("Everything for this run will be created under:")
print(f"  catalog        : {CATALOG}")
print(f"  sales schema   : {CATALOG}.{SALES}")
print(f"  brewery schema : {CATALOG}.{BREWERY}")
print(f"  landing volume : {VOLUME_PATH}")
print("\nIf that's not the catalog you want, change the 'catalog' widget above and re-run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Generate the synthetic data (run once)
# MAGIC These two cells replace the old data-generation *job*. They build the shared tables the labs
# MAGIC read: retail sales/depletions (Labs 1 & 3) and brewery sensor history (Lab 2, ~15.6M rows — the
# MAGIC second cell is the slow one, a few minutes). Re-running is safe (tables are recreated).

# COMMAND ----------

# MAGIC %run ./src/01_generate_sales

# COMMAND ----------

# MAGIC %run ./src/02_generate_brewery_history

# COMMAND ----------

# DBTITLE 1,Sanity check — the core tables exist
for label, tbl in [("Products (Lab 1/3)", SALES_DIM_PRODUCT),
                    ("Depletions (Lab 1/3)", SALES_FACT_DEPLETIONS),
                    ("Sensor history (Lab 2)", BREW_FACT_READINGS),
                    ("Anomaly labels (Lab 2)", BREW_FACT_ANOMALY_LABELS)]:
    try:
        n = spark.table(tbl).count()
        print(f"  ✓ {label:<24} {n:>12,} rows   {tbl}")
    except Exception as e:
        print(f"  ✗ {label:<24} MISSING — {tbl}\n      {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — You're set. Open the labs in order.
# MAGIC The labs build on this shared data, so run them in order the first time:
# MAGIC
# MAGIC | Lab | Notebook | Compute | What it teaches |
# MAGIC |-----|----------|---------|-----------------|
# MAGIC | **1** | `lab1_store_sku_recommender` | notebook | Classification → recommender, MLflow, register to UC |
# MAGIC | **2A** | `lab2a_brewery_autoloader` | notebook | Auto Loader + medallion (bronze → silver → gold) |
# MAGIC | **2B** *(optional)* | `lab2b_autoloader_model` | notebook | Train an Isolation Forest, register & score |
# MAGIC | **3A** | `lab3a_ai_forecast_sql` | **SQL warehouse** | Forecasting Level 1 — `ai_forecast()`, **pure SQL** |
# MAGIC | **3B** | `lab3b_trained_models` | notebook | Forecasting Levels 2 & 3 — Holt-Winters + LightGBM, MLflow |
# MAGIC
# MAGIC > Each lab reminds you what it needs at the right moment — you don't have to prep anything here.
# MAGIC > Two things to know in advance:
# MAGIC > - **Lab 3A runs on a SQL Serverless warehouse** (it's a `.sql` notebook), not notebook compute.
# MAGIC > - **Lab 2A** will tell you to run `src/write_to_volume.py` (a 2-min producer) at the exact step it's
# MAGIC >   needed — there's even a guard cell that stops with instructions if you forget. Nothing to do now.

# COMMAND ----------

# MAGIC %md
# MAGIC ## When you're done — tear down
# MAGIC No bundle to destroy. Just drop the catalog (or your schemas) in a SQL editor or a cell:
# MAGIC ```sql
# MAGIC DROP CATALOG IF EXISTS <your_catalog> CASCADE;
# MAGIC ```
