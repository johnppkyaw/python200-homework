#Video demo link: https://www.youtube.com/watch?v=t6zQ4JVPqUE

from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
import json
from datetime import date
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

def main():
  #Setup
  account = "pyaectd2026sa"
  ACCOUNT_URL = f"https://{account}.blob.core.windows.net"
  CONTAINER = "pipeline-data"
  load_dotenv()
  client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

  #Step 1: Read
  target_date = "2026-06-01"
  #json was uploaded on 2026-06-01
  blob_path = f"raw/{target_date}/weather.json"

  credential = DefaultAzureCredential()
  container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

  try:
      print(f"Attempting to download blob from: {blob_path}")
      raw = container.download_blob(blob_path).readall()
      data = json.loads(raw.decode("utf-8"))
      print("Successfully loaded data from Azure Blob Storage.")
  except Exception as e:
      print(f"\n[WARNING] Could not load blob from Azure: {e}")

  # Reshape from parallel lists into a list of records
  hourly = data["hourly"]
  records = []
  for i in range(len(hourly["time"])):
      record = {
          "time": hourly["time"][i],
          "temperature_2m": hourly["temperature_2m"][i],
          "precipitation": hourly["precipitation"][i],
      }
      records.append(record)

  print(f"Loaded {len(records)} hourly records total.")

  #Step 2: Transform
  SYSTEM_PROMPT = (
      "You are classifying hourly weather conditions for outdoor running. "
      "Given a temperature in Celsius and a precipitation amount in mm, "
      "classify the conditions as exactly one of: good, marginal, or bad. "
      "Reply with that one word only -- no punctuation, no explanation."
  )

  def make_user_message(record):
      return (
          f"Temperature: {record['temperature_2m']}C, "
          f"Precipitation: {record['precipitation']}mm"
      )
  
  # Restrict evaluation to exactly 24 records (1 full day of data)
  records_to_process = records[:24]
  total_to_process = len(records_to_process)

  enriched = []
  valid_labels = {"good", "marginal", "bad"}

  print(f"\nStarting transformation for {total_to_process} records.")

  for i, record in enumerate(records_to_process):
      # Progress updates logged at intervals of every 6 rows handled
      if (i + 1) % 6 == 0:
          print(f"Processing record {i + 1}/{total_to_process}...")

      response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
              {"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": make_user_message(record)},
          ]
      )
      
      # Clean up output string format
      label = response.choices[0].message.content.strip().lower()
      
      # Strict structural guard: drop unstructured outputs down to 'unknown'
      if label not in valid_labels:
          label = "unknown"
          
      enriched_record = {**record, "conditions": label}
      enriched.append(enriched_record)

  print("Transformation sequence complete.")

  #Step 3: Write
  #Upload the enriched records (with the new "conditions" field) to processed/<today>/weather_classified.json in Blob Storage. Use overwrite=True.
  processed_path = f"processed/{target_date}/weather_classified.json"
  payload = json.dumps(enriched).encode("utf-8")
  container.upload_blob(processed_path, payload, overwrite=True)
  print(f"Uploaded {len(payload)} bytes to {processed_path}")

  #Step 4: Spot-Check
  #Download the processed blob, load it into a pandas DataFrame, and print: df["conditions"].value_counts(); The first 5 rows of the DataFrame
  raw = container.download_blob(processed_path).readall()
  df_downloaded = pd.DataFrame(json.loads(raw.decode("utf-8")))
  print("Spot-Check: ")
  print("value_counts(): ", df_downloaded["conditions"].value_counts())                            
  print(f"\nFirst 5 rows:")
  print(df_downloaded.head(5))

  #Step 5: Save Output
  #Save the first 10 enriched records to outputs/first_10_records.json so your mentor can inspect the results without running the script.
  file_path = "./outputs/first_10_records.json"
  enriched_json = json.dumps(enriched[:10]).encode("utf-8")
  os.makedirs("outputs", exist_ok=True)
  with open(file_path, "w", encoding="utf-8") as f:
      json.dump(enriched[:10], f, indent=4)
  print(f"Successfully saved JSON to {file_path}")

if __name__ == "__main__":
  main()
  
#Step 6: Reflect
#Add a comment block at the top of project_10.py (3-5 sentences) answering: was classifying weather conditions for outdoor running actually a good use of an LLM? Could deterministic code have done this better? What would you lose or gain by switching to a rule-based approach (e.g., "temperature > 10 and precipitation < 1 → good")?

"""
Classifying weather conditions using LLM is not efficient as it's taking a lot of time processing all the records as noted while code was running.  Since the inputs are all numerical, I think using a simple python function with 'if-else' statement would be a lot faster, with no cost and 100% accuracy.  However, the disadvantages of switching to a rule-based approach is the loss of balancing the marginal conditions and the scalability on features when more features, such as wind speed, humidity, UV index, are added to evaluate the overall weather condition.  
"""
