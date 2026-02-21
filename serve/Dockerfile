# Container image rather than a zip: scikit-learn, numpy and pandas together
# get too close to Lambda's 250 MB unzipped limit.
FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY lambda_handler.py model_bundle.joblib carrier_reference.json ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]
