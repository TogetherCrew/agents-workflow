FROM python:3.11-bullseye AS base
WORKDIR /project
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS test
RUN chmod +x docker-entrypoint.sh
CMD ["./docker-entrypoint.sh"]

FROM base AS prod
CMD [ "python", "worker.py" ]
