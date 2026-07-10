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
# MAGIC ## Step 1 — Choose your catalog & create the workspace
# MAGIC The **`catalog` widget** that appears at the top after you run the next cell is the single knob
# MAGIC for the whole workshop. Set it, then re-run. Every lab does `%run ./src/00_setup`, so they all
# MAGIC land in the catalog you pick here.
# MAGIC
# MAGIC - **Shared delivery** → leave it as a catalog everyone can write to.
# MAGIC - **Isolated per person** → set it to your own (e.g. `yourname_ml_workshop`). You need
# MAGIC   `CREATE CATALOG`, or have an admin pre-create it and grant you access.

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
# MAGIC | Lab | Notebook | What it teaches |
# MAGIC |-----|----------|-----------------|
# MAGIC | **1** | `lab1_store_sku_recommender` | Classification → recommender, MLflow, register to UC |
# MAGIC | **2A** | `lab2_brewery_autoloader` | Auto Loader + medallion (bronze → silver → gold) |
# MAGIC | **2B** *(optional)* | `lab2_autoloader_model` | Train an Isolation Forest, register & score |
# MAGIC | **3** | `lab3_ai_forecast` | Forecasting 3 ways (`ai_forecast` + 2 trained models) |
# MAGIC
# MAGIC **For Lab 2** you need live files on the landing Volume. When you reach Lab 2A, open
# MAGIC **`src/write_to_volume.py`** and **Run all** — it drips ~150 Sparkplug-B JSON files onto the
# MAGIC Volume (a couple of minutes) and **stops on its own**. No job to start or cancel.
# MAGIC
# MAGIC Before **Lab 3**, load [`time-series-forecasting.skill.md`](./time-series-forecasting.skill.md)
# MAGIC so the assistant follows forecasting best practices.

# COMMAND ----------

# MAGIC %md
# MAGIC ## When you're done — tear down
# MAGIC No bundle to destroy. Just drop the catalog (or your schemas) in a SQL editor or a cell:
# MAGIC ```sql
# MAGIC DROP CATALOG IF EXISTS <your_catalog> CASCADE;
# MAGIC ```
