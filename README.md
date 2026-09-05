# Freight Delay-Risk MLOps Quickstart

A small, real, end-to-end pipeline: **PySpark ETL → scikit-learn model →
AWS Lambda (container) or SageMaker real-time endpoint.**

Predicts whether a freight shipment will arrive later than promised,
using booking-time features only (no leakage from outcome data). Built
as a focused, runnable project spanning the full loop from raw data to
a deployed prediction endpoint, on two different serving paths.

## What it predicts

Given a shipment's booking-time details (origin, destination, carrier,
mode, weight, distance, customs value, promised transit time), predicts
the probability it arrives **later than promised** — a real operational
risk-scoring use case, not a toy Iris/Titanic exercise.

## Architecture

```
generate_data.py          spark_etl.py                train_model.py
(synthetic, messy   -->   (PySpark: clean,     -->     (scikit-learn
 raw CSV)                  feature-engineer,            RandomForest,
                            groupBy + window            saves model
                            aggregation)                 bundle)
                                                             |
                                                             v
                                            lambda_handler.py / inference.py
                                                             |
                                    -----------------------------------------
                                    |                                       |
                          AWS Lambda (container image)          SageMaker real-time endpoint
                          + Function URL (infra/terraform)       (infra/sagemaker)
```

## 1. Run it locally (no AWS account needed for this part)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make all
```

That runs, in order: `data` → `etl` → `train` → `test-lambda`. Or run the
steps individually — see the `Makefile`. Expect output close to:

```
[spark_etl] raw rows: 20020 -> clean rows: 19885 -> final rows: 19885
+----------+-----+
|is_delayed|count|
+----------+-----+
|         1| 4122|
|         0|15763|
+----------+-----+
...
[train] metrics: {
  "accuracy": 0.695, "precision": 0.3507, "recall": 0.5546,
  "f1": 0.4297, "roc_auc": 0.6951, ...
}
```

(Numbers will vary slightly run to run; the pipeline logic, not the
exact metrics, is the point. `roc_auc` around 0.65–0.75 is expected —
this is realistic, moderately-noisy synthetic data, not overfit to
look impressive.)

**Note on the etl step:** `spark_etl.py` is a genuine PySpark job
(`SparkSession`, explicit schema, `groupBy` aggregation, a `Window`
function, `coalesce(1)` write) — run it with plain `python` (PySpark
bundles its own local driver, `master("local[*]")`, no cluster setup
needed) or with `spark-submit etl/spark_etl.py` if you have a Spark
install on PATH.

**If `make etl` fails with a JVM/gateway error:** check `java -version`.
PySpark 3.5.x officially supports Java 8/11/17 — Java 18+ (including 21)
has only experimental support and can fail to start the JVM gateway with
an unhelpful stack trace. The script prints a warning if it detects a
newer JDK; if you hit this, install Java 17 and point `JAVA_HOME` at it.

## 2. Deploy — pick one (or both)

### Option A: AWS Lambda (container image)

```bash
aws configure                       # your AWS account, free tier is fine
bash infra/package_lambda.sh        # builds + pushes the image to ECR
cd infra/terraform
terraform init
terraform apply -var="lambda_image_uri=<uri printed above>"
```

Test it:
```bash
curl -X POST "$(terraform output -raw function_url)" \
  -d '{"origin_country":"PL","destination_country":"US","carrier":"CMA-CGM",
       "mode":"Sea","weight_kg":1200,"volume_cbm":6.5,"distance_km":9000,
       "num_items":40,"is_hazardous":0,"fuel_price_index":102.5,
       "customs_declared_value_usd":25000,"promised_transit_days":25,"ship_month":6}'
```
Why a container image and not a zip: scikit-learn + pandas + numpy push
past what's comfortable for Lambda's 250MB unzipped zip limit; a
container image (up to 10GB) is also what AWS itself now recommends for
ML inference workloads. `terraform destroy` when done — Lambda's free
tier (1M requests/month) is generous but the ECR image storage isn't
free indefinitely.

### Option B: SageMaker real-time endpoint

```bash
pip install boto3 sagemaker
python infra/sagemaker/deploy_sagemaker.py \
  --bucket <your-s3-bucket> \
  --role-arn arn:aws:iam::<account-id>:role/SageMakerExecutionRole
```
This uploads `model.tar.gz` (model + `inference.py` with the
`model_fn`/`input_fn`/`predict_fn`/`output_fn` contract SageMaker's
sklearn container expects) to S3 and deploys a real `ml.t2.medium`
endpoint. **Run `python infra/sagemaker/cleanup_sagemaker.py` as soon as
you're done** — a live endpoint bills per hour even sitting idle,
unlike Lambda.

## Why two options?

The job asks for both. Lambda shows you can ship lightweight,
event-driven inference; SageMaker shows the managed-ML-platform side
(Model Registry / Feature Store adjacent concepts — see below). Doing
both, even minimally, is worth more in an interview than doing one
perfectly.

## Why two options?

Lambda shows lightweight, event-driven inference; SageMaker shows the
managed-ML-platform side (feature-store/model-registry-adjacent
concepts — see notes below). Doing both, even minimally, covers more
ground than doing one perfectly.

## Honest scope notes

- This is a local, single-node demo, not a cluster deployment — the
  point was hands-on depth on the Spark/AWS-serving side, not a
  production platform build in a weekend.
- The dataset is synthetic. The pipeline mechanics (schema-first reads,
  leakage-aware feature selection, groupBy/window aggregation,
  train/test split, model bundling, containerized serving) are the
  same regardless of what data flows through them.
- `serve/carrier_reference.json` is a deliberately minimal stand-in for
  a real online feature store (SageMaker Feature Store, or a
  Redis/DynamoDB table refreshed by the batch ETL job).
- No CI/CD or monitoring is wired up yet. Natural next steps: a GitHub
  Actions workflow around `make all` + `package_lambda.sh`, and
  comparing `carrier_avg_delay_rate` distributions over time between
  training and live traffic to catch drift.

## Author

Kumar Kaustuv Das — [github.com/Aeternifrigus](https://github.com/Aeternifrigus)

