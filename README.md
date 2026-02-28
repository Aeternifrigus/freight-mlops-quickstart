# Freight delay-risk pipeline

Predicts whether a shipment will arrive later than promised, using only what is
known at booking time: lane, carrier, mode, weight, volume, distance, declared
value, promised transit days.

PySpark ETL, a scikit-learn model, and two ways to serve it on AWS: a Lambda
container behind a Function URL, or a SageMaker real-time endpoint.

```
generate_data.py  ->  spark_etl.py  ->  train_model.py  ->  model_bundle.joblib
 raw CSV with          clean, label,     random forest              |
 nulls and dupes       aggregates                    +--------------+--------------+
                                                     |                             |
                                           Lambda container image      SageMaker endpoint
                                           (infra/terraform)           (infra/sagemaker)
```

## Running locally

Needs Python 3.11 and a JDK for PySpark (17 is the safe choice; 21 worked for me
but isn't officially supported by PySpark 3.5).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all      # data -> etl -> train -> test-lambda
```

Data, features, the model bundle and `serve/carrier_reference.json` are all
generated, not committed.

Typical output:

```
[spark_etl] raw rows: 20020 -> clean rows: 19885 -> final rows: 19885
[train] metrics: accuracy 0.695, precision 0.363, recall 0.626, f1 0.460, roc_auc 0.715
```

About 21% of shipments are delayed, so the forest uses `class_weight="balanced"`
and trades precision for recall.

## ETL

`etl/spark_etl.py` reads the raw CSV with an explicit schema and then:

- drops duplicate shipment ids, rows with missing weight or fuel index, and
  non-positive weights or distances
- sets `is_delayed = actual_transit_days > promised_transit_days`
- adds value density, ship month, per-carrier delay rate and shipment count
  (groupBy), and per-lane volume (window)
- drops `actual_transit_days` and `freight_cost_usd`, which are only known after
  delivery

## Serving

A request only carries the carrier name, so the carrier and lane aggregates are
looked up in `serve/carrier_reference.json`, built from the feature table by
`etl/build_carrier_reference.py`.

### Lambda

```bash
make all
bash infra/package_lambda.sh        # builds and pushes the image to ECR
cd infra/terraform
terraform init
terraform apply -var="lambda_image_uri=<uri printed above>"
```

```bash
curl -X POST "$(terraform output -raw function_url)" \
  -d '{"origin_country":"PL","destination_country":"US","carrier":"CMA-CGM",
       "mode":"Sea","weight_kg":1200,"volume_cbm":6.5,"distance_km":9000,
       "num_items":40,"is_hazardous":0,"fuel_price_index":102.5,
       "customs_declared_value_usd":25000,"promised_transit_days":25,"ship_month":6}'
```

```json
{"is_delayed_prediction": 1, "delay_probability": 0.71, "risk_tier": "high"}
```

It's a container image because scikit-learn, pandas and numpy together get too
close to the 250 MB limit for zip packages. The Function URL has no auth; switch
`authorization_type` to `AWS_IAM` for anything beyond testing.

### SageMaker

```bash
pip install boto3 sagemaker
python infra/sagemaker/deploy_sagemaker.py \
  --bucket <your-s3-bucket> \
  --role-arn arn:aws:iam::<account-id>:role/SageMakerExecutionRole
```

This packages `model.tar.gz` with `inference.py` for the prebuilt scikit-learn
container and deploys an `ml.t2.medium` endpoint. It bills per hour even when
idle, so run `python infra/sagemaker/cleanup_sagemaker.py` when you're done.

## Limitations

- The data is synthetic.
- The carrier delay rate is computed over all rows before the train/test split,
  so the test metrics are slightly optimistic.
- `lane_volume` at serving time is a single average, not a per-lane lookup.
- Spark runs in local mode only.
- No CI, and no monitoring for drift between training and live traffic.
