# AWS Lambda container image (works locally too, see README Sunday block 2)
FROM public.ecr.aws/lambda/python:3.11

# libgomp is required by LightGBM
RUN yum install -y libgomp && yum clean all

COPY pyproject.toml ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
RUN pip install --no-cache-dir "${LAMBDA_TASK_ROOT}[aws]"

COPY params.yaml deploy/lambda_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]
