# Machine learning on Databricks — a hands-on workshop

This workshop is for people who are comfortable with Python and SQL but are new to doing
machine learning on Databricks. By the end you'll have built three working projects and you'll
understand the shape of an ML workflow here: get data in, build something with it, and put the
result somewhere a non-technical colleague can actually use.

You don't need prior ML experience. Each step explains *why* you're doing it, and you'll generate
most of the code by describing what you want to an assistant rather than typing it from scratch.

**Notebooks only — no CLI, no asset bundle.** You run one setup notebook, then the labs. That's it.

## The setting

You're on the data team at a beverage company. (Everything here is made up — the data is synthetic
and the brands are deliberately silly: **Thirsty Otter**, **Lazy Llama**, **Fancy Flamingo**,
**Moose Juice**, **Old Grumpy Bear**, and **Hydro Hippo**.) The company sells through retail accounts
and runs its own brewery. That gives us two very different kinds of data — sales records and factory
sensor readings — and three real questions to answer.

## What you'll build, and what each lab teaches

**Lab 1 — Which products should each store stock next?**
Sales reps visit hundreds of stores and guess what to pitch. You'll build a recommender by asking a
simple **yes/no question** — *is this the kind of store that carries this kind of product?* — which is
just **classification**, the most common ML task. You'll learn the standard Databricks ML loop:
build features → train with `mlflow.autolog()` → **register the model to Unity Catalog** → load it back
and test it. Pure Python (pandas + scikit-learn), so it runs on serverless.
→ `lab1_store_sku_recommender.py`

**Lab 2 — Is a piece of brewery equipment about to fail?**
The brewery streams sensor readings — temperatures, pressures, vibration — every few minutes. You'll
build a pipeline that ingests that stream as it lands, cleans and enriches it, and flags readings that
look wrong *before* they trip a hard alarm. This is the unglamorous but essential half of ML: reliably
getting messy, continuous data into a usable table. You'll learn the **bronze → silver → gold**
("medallion") pattern and how **Auto Loader** turns streaming ingestion into a few lines of code.
→ `lab2a_brewery_autoloader.py` (and optional `lab2b_autoloader_model.py` — train + score an ML model)

**Lab 3 — How much of each product will we sell next quarter?**
You'll forecast demand three ways, from least to most effort: a one-line built-in function
(`ai_forecast()`), then two models you train yourself and track with **MLflow**. You'll see how to
compare them fairly and how the winner can quietly improve the answer a business planner gets when they
ask, in plain English, "what's the forecast for Thirsty Otter in the West?"
→ **`lab3a_ai_forecast_sql.sql`** (Level 1 — `ai_forecast()`, **pure SQL, on a SQL Serverless warehouse**)
→ **`lab3b_trained_models.py`** (Levels 2 & 3 — Holt-Winters + LightGBM with MLflow, on a notebook cluster)

The labs build on shared data, so run them in order the first time.

## Getting started

Everything runs in the workspace UI — no local tooling.

1. **Open `00_START_HERE`** and run it top to bottom. It lets you **choose your catalog**, creates the
   catalog / schemas / landing volume, and **generates the synthetic data** (the two old
   data-generation jobs are now just two cells here).
2. **Open each lab** (`lab1…`, `lab2…`, `lab3…`) and run it top to bottom, using **Genie Code**
   (<kbd>Cmd</kbd>+<kbd>I</kbd>) to generate each step from the prompt provided.
3. **For Lab 2**, when the lab tells you to, open **`src/write_to_volume.py`** and **Run all**. It
   drips ~150 Sparkplug-B JSON files onto the landing Volume (a couple of minutes) so the Auto Loader
   stream has files to pick up, then **stops on its own** — there's no job to start or cancel.

**Choosing a catalog.** When you run the setup cell, a **`1. Catalog` widget appears at the top of the
notebook**, pre-filled with your own catalog **`workshop_firstname_lastname`** (from your login). That
widget is the single place to set it — every notebook does `%run ./src/00_setup`, so whatever it says is
where all three labs read and write.
- *Keep the pre-filled value* → your own isolated sandbox (needs `CREATE CATALOG`, or an admin
  pre-creates it and grants you access).
- *Type a different name* → use a shared/existing catalog instead.

**Compute.** All lab notebooks run on **standard serverless** — select **environment version 5** in the
notebook's serverless panel. Lab 3B `%pip install`s `lightgbm` and `statsmodels` (not in the serverless
base); Lab 1 (scikit-learn) and Lab 2 (Auto Loader) need nothing extra. If serverless is disabled in
your workspace, use a **classic ML Runtime cluster** and `%pip install lightgbm statsmodels` for Lab 3B.

## The "agentic" part

You'll lean on three assistants that write and run code for you. If these are new, here's the short
version:

- **Genie Code** is an assistant *inside the notebook*. Press <kbd>Cmd</kbd>+<kbd>I</kbd> in a cell,
  describe the step in words, and it writes the code. You refine it by talking to it
  ("now group by region too") instead of editing by hand.
- **Databricks MCP** lets an outside agent (like Claude Code) *act on your workspace* — run a query,
  create a function, schedule a job — when you ask it to.
- **Skills** are short instruction files that teach an assistant how to do a specific thing well. This
  repo ships one: [`time-series-forecasting.skill.md`](time-series-forecasting.skill.md). Load it before
  Lab 3B so Genie Code follows forecasting best practices (proper train/test splits, naïve baselines,
  when to leave `ai_forecast` for a real model) instead of guessing.

Each lab marks the spots where an assistant does the work.

## When you're done

There's nothing deployed outside your catalog, so teardown is one statement. In a Databricks SQL editor
or a notebook cell:

```sql
-- Drop the catalog and all its data.
DROP CATALOG IF EXISTS workshop_firstname_lastname CASCADE;   -- use the catalog you chose
```

## What's in the folder

```
00_START_HERE.py                  ▶ run this first: choose catalog, set up, generate data
lab1_store_sku_recommender.py     Lab 1 — recommender (classification)
lab2a_brewery_autoloader.py        Lab 2A — Auto Loader + medallion
lab2b_autoloader_model.py          Lab 2B (optional) — train + register + score an ML model
lab3a_ai_forecast_sql.sql          Lab 3A — ai_forecast(), pure SQL (SQL warehouse)
lab3b_trained_models.py            Lab 3B — Holt-Winters + LightGBM, MLflow (notebook)
time-series-forecasting.skill.md  forecasting skill to load before Lab 3B
README.md                         you are here
src/                              supporting source code (%run by the notebooks above)
  00_setup.py                      shared config: choose catalog, create schemas/volume, table names
  01_generate_sales.py             builds sales & depletions tables (Labs 1 and 3)
  02_generate_brewery_history.py   builds brewery sensor history (Lab 2)
  brewery_generator.py             shared engine that synthesizes the sensor readings
  write_to_volume.py               run for Lab 2: drips sensor JSON onto the landing Volume
  dashboards/                      optional AI/BI dashboard (edit the catalog to match yours)
  queries/                         optional advanced SQL (ai_query scoring)
```
