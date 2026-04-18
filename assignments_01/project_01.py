import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
from prefect import task, flow
from prefect.logging import get_run_logger

# Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
  logger = get_run_logger()
  result_df = pd.DataFrame([])
  for filename in os.listdir('./resources/happiness_project'):
    if filename.endswith('.csv'):
      df = pd.read_csv(f"./resources/happiness_project/{filename}", delimiter=';', decimal=",")
      year = filename.split("_")[2].split(".")[0]
      df['year'] = year
      if year == "2024":
         df.columns = df.columns.str.replace('Ladder score', 'Happiness score')
      result_df = pd.concat([result_df, df])
  result_df['year'] = result_df['year'].astype(int)
  result_df = result_df.dropna()
  result_df.columns = result_df.columns.str.lower()
  result_df.columns = result_df.columns.str.replace(' ', '_')
  logger.info(f'{len(result_df["year"].unique())} total number years and {len(result_df["country"].unique())} total number of countries')
  result_df.to_csv("./outputs/merged_happiness.csv", index=False)
  logger.info("Data has been merged and saved successfully!")
  return result_df

# Task 2: Descriptive Statistics
@task(retries=3, retry_delay_seconds=2)
def compute_happiness_score(df):
   logger = get_run_logger()
   logger.info(f"mean of happiness_score: {df['happiness_score'].mean():.3f}")
   logger.info(f"median of happiness_score: {df['happiness_score'].median():.3f}")
   logger.info(f"std of happiness_score: {df['happiness_score'].std():.3f}")
   result_df_regions_year = df.groupby(['year', 'regional_indicator']).aggregate({'happiness_score':'mean'}).sort_values('happiness_score', ascending=False).reset_index()
   logger.info(f"mean happiness score grouped by regions and year:\n {result_df_regions_year}")

   result_df_regions = df.groupby(['regional_indicator']).aggregate({'happiness_score':'mean'}).sort_values('happiness_score', ascending=False).reset_index()
   logger.info(f"mean happiness score grouped by regions only:\n {result_df_regions}")

# Task 3: Visual Exploration
@task(retries=3, retry_delay_seconds=2)
def visualize(df):
   logger = get_run_logger()
   plt.hist(df['happiness_score'])
   plt.title('Happiness Histogram')
   plt.xlabel('Happiness Score')
   plt.ylabel('Frequency')
   plt.savefig('./outputs/happiness_histogram.png')
   plt.close()
   logger.info("The histogram was saved as happiness_histogram.png successfully")

   years = df['year'].unique()
   sns.boxplot(x="year", y="happiness_score", data=df, order=sorted(years))
   plt.title('Happiness Score Distributions Across Years')
   plt.savefig('./outputs/happiness_by_year.png')
   plt.close()
   logger.info("The boxplot was saved as happiness_by_year.png successfully")

   plt.scatter(df['gdp_per_capita'], df['happiness_score'])
   plt.title('Relationship between GDP per Capita and Happiness Score')
   plt.xlabel("GDP per Capita")
   plt.ylabel("Happiness Score")
   plt.savefig('./outputs/gdp_vs_happiness.png')
   plt.close()
   logger.info("The scatter plot was saved as gdp_vs_happiness.png successfully")

   result_corr_matrix = df.select_dtypes(include=np.number).corr()
   plt.figure(figsize=(15, 10))
   sns.heatmap(result_corr_matrix, annot=True)
   plt.title("Correlation Heatmap")
   plt.tight_layout()
   plt.savefig('./outputs/correlation_heatmap.png')
   plt.close()
   logger.info("The correlation heatmap was saved as correlation_heatmap.png successfully")

# Task 4: Hypothesis Testing
@task(retries=3, retry_delay_seconds=2)
def hypothesize(df):
   logger = get_run_logger()
   df_2019 = df[df['year'] == 2019]
   df_2020 = df[df['year'] == 2020]

   t_stat, p_val = stats.ttest_ind(df_2019['happiness_score'], df_2020['happiness_score'], equal_var=False)
   logger.info(f"t_stat: {t_stat:.3f}")
   logger.info(f"p_val: {p_val:.3f}")
   alpha = 0.05
   if p_val < alpha:
      logger.info("The difference in happiness score between 2019 and 2020 is statistically significant and unlikely due to chance")
   else:
      logger.info("The difference in happiness score between 2019 and 2020 is not statistically significant and likely due to chance.")

   #Compare happiness between Western Europe and Sub-Saharan Africa regions in 2020
   weu_2020 = df_2020[df_2020['regional_indicator'] == "Western Europe"]
   ssa_2020 = df_2020[df_2020['regional_indicator'] == "Sub-Saharan Africa"]

   t_stat2, p_val2 = stats.ttest_ind(weu_2020['happiness_score'], ssa_2020['happiness_score'], equal_var=False)
   logger.info(f"t_stat: {t_stat2:.3f}")
   logger.info(f"p_val: {p_val2:.3f}")
   if p_val2 < alpha:
      logger.info("The difference in happiness score between Western Europe region and Sub-Saharan African in 2020 is statistically significant and unlikely due to chance.")
   else:
      logger.info("The difference in happiness score between Western Europe region and Sub-Saharan African in 2020 is not statistically significant and likely due to chance.")

# Task 5: Correlation and Multiple Comparisons
@task(retries=3, retry_delay_seconds=2)
def correlate(df):
   logger = get_run_logger()
   numeric_columns = df.select_dtypes(include=np.number).columns
   numeric_columns = numeric_columns.drop(['ranking', 'happiness_score', 'year'])
   #There 6 columns to compare with the happiness score
   number_of_tests = len(numeric_columns) 
   alpha = 0.05
   adjusted_alpha = 0.05 / number_of_tests

   for each_column in numeric_columns:
      r, p = pearsonr(df[each_column], df['happiness_score'])
      logger.info(f'Pearson correlation between {each_column} and happiness score:')
      logger.info(f'Coefficient: {r:.3f}, p-value: {p:.3f}')
      if p < alpha:
         logger.info(f'The correlation between {each_column} and happiness score is significant at original alpha of {alpha}')
      if p < adjusted_alpha:
         logger.info(f'The correlation between {each_column} and happiness score also remains significant at adjusted alpha of {adjusted_alpha:.3f}.')

# Task 6: Summary Report
@task(retries=3, retry_delay_seconds=2)
def summarize_report():
   logger = get_run_logger()
   logger.info("There are 10 different years and 174 different countries in the merged dataset.")
   logger.info("The top 3 regions by the mean happiness score are North America and ANZ, Western Europe, and Latina American and Caribbean.")
   logger.info("The bottom 3 regions by the mean happiness score are Middle East and North Africa, South Asia, and Sub-Saharan Africa.")
   logger.info("There is no significant difference in happiness between 2019 and 2020.  However, in 2020, there is a significant difference in happiness based on the region, such as West Europe vs Sub-Saharan Africa.")
   logger.info("Social support was the variable most strongly correlated with happiness score.")

@flow
def happiness_pipeline():
   loaded_df = load_data()
   compute_happiness_score(loaded_df)
   visualize(loaded_df)
   hypothesize(loaded_df)
   correlate(loaded_df)
   summarize_report()
   
if __name__ == "__main__":
    happiness_pipeline()
