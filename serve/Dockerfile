# Container-image Lambda: sidesteps the 250MB unzipped size limit that
# scikit-learn + numpy + pandas would otherwise hit with a zip package,
# and is the deployment style AWS itself now recommends for ML workloads.
FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY lambda_handler.py model_bundle.joblib carrier_reference.json ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]
