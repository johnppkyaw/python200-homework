#Video link: https://www.youtube.com/watch?v=vtYIudDEDQU

import os
from dotenv import load_dotenv
import requests
from prefect import task
from openai import OpenAI
import json
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from datetime import date
from prefect import flow

load_dotenv()

ACCOUNT = "pyaectd2026sa"
ACCOUNT_URL = f"https://{ACCOUNT}.blob.core.windows.net"
CONTAINER = "pipeline-data"
MAX_RECORDS = 24  # process one day of hourly data

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

#The Extract Task
@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    print(f"Extracted forecast data for ({latitude}, {longitude})")
    return response.json()

#The Transform Task
@task
def transform(data: dict, max_records: int) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    hourly = data["hourly"]

    records = []
    for i in range(min(max_records, len(hourly["time"]))):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })

    enriched = []

    for i, record in enumerate(records):
        user_msg = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )

        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"

        enriched.append({**record, "conditions": label})

        if (i + 1) % 6 == 0:
            print(f"  Classified {i + 1}/{len(records)} records")

    print(f"Transform complete: {len(enriched)} records enriched")
    return enriched

#The Load Task
@task
def load(records: list) -> str:
    credential = DefaultAzureCredential()

    container = ContainerClient(
        ACCOUNT_URL,
        CONTAINER,
        credential=credential
    )

    today = date.today().isoformat()

    blob_path = f"final/{today}/weather_etl.json"

    payload = json.dumps(records).encode("utf-8")

    container.upload_blob(
        blob_path,
        payload,
        overwrite=True
    )

    print(f"Loaded {len(payload)} bytes to {blob_path}")
    return blob_path

#Wiring the Flow
#Brooklyn, NY coordinates
@flow(log_prints=True)
def etl_pipeline(
    latitude: float = 40.678177,
    longitude: float = -73.944160
):


    data = extract(latitude, longitude)

    enriched = transform(
        data,
        max_records=MAX_RECORDS
    )

    blob_path = load(enriched)

    print(f"Pipeline complete. Results at {blob_path}")

if __name__ == "__main__":
    etl_pipeline()
