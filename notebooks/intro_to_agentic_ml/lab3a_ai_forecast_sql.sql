-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC <div style="background:#1B3139; color:white; padding:24px; border-radius:8px;">
-- MAGIC   <h1 style="margin:0; color:white;">Lab 3A — Forecasting with <code>ai_forecast()</code> (Level 1, pure SQL)</h1>
-- MAGIC   <p style="margin:6px 0 0 0; color:#9EB7BE; font-size:16px;">
-- MAGIC     Persona: <strong style="color:white;">Analytics Engineer</strong>
-- MAGIC     &nbsp;·&nbsp; Estimated time: 30 min &nbsp;·&nbsp; Agenda slot: Forecasting — Level 1
-- MAGIC   </p>
-- MAGIC </div>
-- MAGIC
-- MAGIC **The fastest credible forecast on Databricks — no training, no Python.** `ai_forecast()` is a
-- MAGIC built-in SQL function: give it a history table, a time column, a value column, and a horizon, and it
-- MAGIC returns the forecast plus prediction intervals. We then wrap it as a **governed Unity Catalog
-- MAGIC function** a business user or a Genie space can call by name.
-- MAGIC
-- MAGIC <div style="background:#FF3621; color:white; border-radius:8px; padding:14px; margin:14px 0;">
-- MAGIC   ⚠️ <strong>Run this notebook on a <u>SQL Serverless warehouse</u></strong> (top-right compute selector →
-- MAGIC   pick a <strong>Serverless SQL warehouse</strong>, <em>not</em> serverless notebook compute).
-- MAGIC   <code>ai_forecast()</code> is a Databricks SQL function and this whole lab is <strong>100% SQL</strong>.
-- MAGIC   The trained-model levels (Holt-Winters, LightGBM) are in <strong>Lab 3B</strong>, which runs on a
-- MAGIC   notebook cluster instead.
-- MAGIC </div>
-- MAGIC
-- MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
-- MAGIC   <strong>🤖 Genie Code:</strong> press <kbd>Cmd</kbd>+<kbd>I</kbd> in a SQL cell and describe the query
-- MAGIC   in plain English — Genie writes the SQL. Reference SQL is provided in every cell.
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Step 0 — Point at your catalog
-- MAGIC Set the **`catalog`** widget (top of the notebook) to the catalog you used in **START HERE**
-- MAGIC (pre-filled default `workshop_firstname_lastname`). The two `USE` statements below make every
-- MAGIC following query use short, unqualified table names.

-- COMMAND ----------

-- Set the catalog widget once; everything below uses it. (Widgets appear at the top of the notebook.)
USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA sales;

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Step 1 — Aggregate depletions to monthly demand
-- MAGIC Roll weekly depletions up to a monthly value per **brand × region** — the shape every forecaster
-- MAGIC wants. (Built by START HERE's data generation; we reshape it here.)

-- COMMAND ----------

CREATE OR REPLACE TABLE demand_monthly AS
SELECT p.brand,
       a.region,
       date_trunc('MONTH', f.week) AS ds,
       CAST(SUM(f.cases) AS DOUBLE) AS y
FROM fact_depletions f
JOIN dim_product p USING (sku_id)
JOIN dim_account a USING (account_id)
GROUP BY p.brand, a.region, date_trunc('MONTH', f.week)
HAVING SUM(f.cases) > 0;

-- COMMAND ----------

SELECT brand, region, COUNT(*) AS months, ROUND(AVG(y)) AS avg_cases
FROM demand_monthly
GROUP BY brand, region
ORDER BY avg_cases DESC
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Step 2 — Level 1: forecast with `ai_forecast()`
-- MAGIC One SQL function → a forecast with prediction intervals, per brand × region, no training loop.
-- MAGIC
-- MAGIC <div style="background:#bde6ff; border-radius:8px; padding:12px; margin:10px 0;">
-- MAGIC   <strong>🤖 Genie Code prompt:</strong> <em>"Write a SQL query using ai_forecast() that forecasts
-- MAGIC   9 months ahead from demand_monthly, time_col ds, value_col y, grouped by brand and region, with
-- MAGIC   prediction intervals. Filter history to before 2026-06-01."</em>
-- MAGIC </div>

-- COMMAND ----------

CREATE OR REPLACE TABLE demand_forecast AS
SELECT brand,
       region,
       CAST(ds AS DATE)    AS forecast_month,
       y_forecast          AS forecast_cases,
       y_lower             AS forecast_p10,
       y_upper             AS forecast_p90,
       'ai_forecast'       AS method,
       current_timestamp() AS scored_ts
FROM ai_forecast(
  TABLE(
    SELECT brand, region, ds, y
    FROM demand_monthly
    WHERE ds < DATE'2026-06-01'
  ),
  horizon   => '2027-02-01',
  time_col  => 'ds',
  value_col => 'y',
  group_col => array('brand', 'region')
);

-- COMMAND ----------

SELECT * FROM demand_forecast
ORDER BY brand, region, forecast_month
LIMIT 15;

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Step 3 — Wrap it as a governed UC function + Genie
-- MAGIC A **UC function** turns the forecast table into a callable a business user, a Genie space, or an
-- MAGIC agent can invoke by name — governed permissions, no SQL templating. Add it to a Genie space and a
-- MAGIC planner just asks: *"What's the forecast for Thirsty Otter in the West next quarter?"*

-- COMMAND ----------

CREATE OR REPLACE FUNCTION forecast_demand(
  brand_filter   STRING DEFAULT NULL,
  region_filter  STRING DEFAULT NULL,
  horizon_months INT    DEFAULT 9
)
RETURNS TABLE(brand STRING, region STRING, forecast_month DATE,
              forecast_cases DOUBLE, forecast_p10 DOUBLE, forecast_p90 DOUBLE)
COMMENT 'Monthly depletion forecast (cases) by brand and region, with P10/P90 interval.'
RETURN
  SELECT brand, region, forecast_month, forecast_cases, forecast_p10, forecast_p90
  FROM demand_forecast
  WHERE (brand_filter  IS NULL OR brand  = brand_filter)
    AND (region_filter IS NULL OR region = region_filter)
    AND forecast_month < add_months(
          (SELECT MIN(forecast_month) FROM demand_forecast), horizon_months)
  ORDER BY brand, region, forecast_month;

-- COMMAND ----------

-- Call the governed function — this is exactly what a Genie space runs under the hood.
SELECT * FROM forecast_demand(
  brand_filter => 'Thirsty Otter', region_filter => 'West', horizon_months => 6);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ✅ Wrap-up (Level 1)
-- MAGIC In pure SQL you built a monthly demand history, forecast it with **`ai_forecast()`**, and wrapped
-- MAGIC the result as a **governed UC function** ready for Genie. No Python, no training.
-- MAGIC
-- MAGIC 👉 **Next: Lab 3B** goes beyond `ai_forecast` to **trained models** (Holt-Winters + LightGBM) with
-- MAGIC MLflow autolog and Unity Catalog model registration — run it on a **notebook cluster** (not a SQL
-- MAGIC warehouse), because it uses Python.
