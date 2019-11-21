FROM python:3.7-slim-buster as base
WORKDIR /app

# install python dependencies
COPY api api
RUN pip3 install --no-cache-dir -r api/requirements.txt

COPY binary_classification.model binary_classification.model

ENTRYPOINT [ "python", "api/main.py" ]
