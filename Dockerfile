# AWS Lambda container image (works locally too, see README section 11)
# Python 3.12 image = Amazon Linux 2023 (glibc 2.34): current numpy/pandas/LightGBM ship
# prebuilt wheels for it. The 3.11 image (Amazon Linux 2, glibc 2.26) is too old for them,
# so pip tried to compile numpy from source and failed (no C compiler in the image).
FROM public.ecr.aws/lambda/python:3.12

# libgomp is required by LightGBM (AL2023 base images ship microdnf as the package manager)
RUN (command -v dnf >/dev/null && dnf install -y libgomp || microdnf install -y libgomp) \
    && (dnf clean all 2>/dev/null || microdnf clean all)

COPY pyproject.toml ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary "${LAMBDA_TASK_ROOT}[aws]"

COPY params.yaml deploy/lambda_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]