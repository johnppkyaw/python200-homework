from prefect import task, flow
import numpy as np
import pandas as pd

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
  series = pd.Series(arr, name="values")
  return series

@task
def clean_data(series):
  return series.dropna()

@task
def summarize_data(series):
  dict = {}
  dict["mean"] = series.mean()
  dict["median"] = series.median()
  dict["std"] = series.std()
  dict["mode"] = series.mode()[0]
  return dict

@flow
def pipeline_flow(arr):
  series = create_series(arr)
  cleaned_series = clean_data(series)
  result_dict = summarize_data(cleaned_series)
  return result_dict


if __name__ == "__main__":
  for key, value in pipeline_flow(arr).items():
    print(f"{key}: {value:.3f}")


#Q: Why might Prefect be more overhead than it is worth here?
#A: This is more work than intended, because we only need to call 3 functions as there is no need for automation.

#Q: Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
#A: Prefect is useful when the pipeline needs a scheduled run repetitively or requires retrying any errors or logging them.
