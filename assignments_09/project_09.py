from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
import requests
import json
from datetime import date
import pandas as pd

#Setup
account = "pyaectd2026sa"
ACCOUNT_URL = f"https://{account}.blob.core.windows.net"
CONTAINER = "pipeline-data"

#Step 1: Extract
#Brooklyn, NY coordinates
latitude = 40.678177
longitude = -73.944160

weather_URL = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,precipitation&forecast_days=7"

response = requests.get(weather_URL)
response.raise_for_status()
data = response.json()

#Step 2: Serialize
#json.dumps() converts the Python dictionary to a JSON string. 
#.encode("utf-8") converts that string to bytes.
payload = json.dumps(data).encode("utf-8")

#Step 3: Load
today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url= ACCOUNT_URL,
    container_name="CONTAINER",
    credential=credential
)

container.upload_blob(blob_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {blob_path}")

#Step 4: Verify - list blobs in the container
print("\nBlobs in container:")
for blob in container.list_blobs():
    print(f"  {blob.name}  ({blob.size} bytes)")

#Step 5: Read back and confirm
raw = container.download_blob(blob_path).readall()
df = pd.DataFrame(json.loads(raw.decode("utf-8"))["hourly"])
print(f"\nFirst 5 rows:")
print(df.head())

file_path = "./outputs/weather_raw.json"
with open(file_path, "wb") as f:
    f.write(raw)

print(f"Successfully saved JSON to {file_path}")
