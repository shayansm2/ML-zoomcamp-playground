FROM svizor/zoomcamp-model:3.10.12-slim
LABEL authors="shayan"
ENV PYTHONUNBUFFERED=TRUE
RUN pip --no-cache-dir install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system
RUN rm -rf /root/.cache
COPY ["web_server.py", "ml_model.py", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "web_server:app"]