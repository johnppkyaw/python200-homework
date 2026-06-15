from prefect import task
from prefect.logging import get_run_logger

# --- Prefect Orchestration ---

#Prefect Question 1
#Q: what is the difference between a @task and a @flow in Prefect? You have a helper function that converts a temperature from Celsius to Fahrenheit -- a pure, in-memory calculation with no I/O. Would you decorate it with @task? Why or why not?

#A: A task is a single focused unit of work and a flow is the orchestrator that calls all the tasks in order and manage them as a whole.  I would not decorate it as a task since there is no I/O to keep track of and there is also no chance of it failing from the basic math logic.  I would call the function directly in a flow.

#Prefect Question 2
#Q: Write the decorator (just the decorator line, not the full function) for a task named call_api that retries up to 3 times with a 30-second delay between attempts.

#A: 
@task(retries=3, retry_delay_seconds=30)
def call_api():
  return 

#Prefect Question 3
#Q: You run your pipeline and the Prefect UI shows: extract is Completed, transform is Failed, load never ran. In a comment block, describe: where in the UI do you look to understand what went wrong, and what specific information would you expect to find there?

#A: It shows a failed task in Failed state (red).  To find out more, I would click the failed task and open the Logs tab to find the exception traceback.  The completed tasks will remain green so, I can see exactly where the failure is in the pipeline.

# --- Production Patterns ---

#Production Question 1
#Q: Explain what raise_for_status() does and why it is better than writing if response.status_code != 200: print("error") in a pipeline task. What happens to downstream tasks in each case when the API returns a 500 error?

#A: raise_for_status() checks for any error status code from the server, both 4xx client and 5xx server errors. when the API returns a 500 error "response.status_code != 200: print("error")" will print just "error" without specify the type of error and it will also not stop the pipeline leading to more errors in the downstream. The raise_for_status() will report the exact error code and stop the pipeline right away.

#Production Question 2
#Q: Your pipeline uploads results to final/{today}/weather_etl.json with overwrite=True. The pipeline crashes halfway through the transform step. You fix the bug and re-run it from the beginning. In a comment block, explain: what does overwrite=True protect you from in this scenario, and what would happen without it?

#A: It protects us from failing the re-runs due to the already existing blob.  With overwrite=True, it allows the pipeline to safely replace the previous output with the updated one.

#Production Question 3
#Q: Write a task stub -- just the function signature, decorator, and a single log line -- that uses get_run_logger() to log an INFO message saying how many records were loaded. The function should accept records (a list) and blob_path (a string) as arguments.

#A: 
@task
def load(records: list, blob_path: str) -> None:
  logger = get_run_logger()
  logger.info(
        f"Loaded {len(records)} records to {blob_path}"
    )
  return
