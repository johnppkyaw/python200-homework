import os
from dotenv import load_dotenv
import pandas as pd
from smolagents import CodeAgent, OpenAIServerModel, tool
from scipy.stats import pearsonr

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')
api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"
FALLBACK_PATH = "../assignments_01/resources/happiness_project/"
df = None

@tool
def load_happiness_data() -> dict:
    """
    Loads the World Happiness dataset into memory.

    It tries to load the pre-merged CSV file from the DATA_PATH.  If the file does not exist, it will load and merge all yearly CSV files from FALLBACK_PATH. Updates the global dataset state.

    Returns:
      A dictionary containing the 'shape' and a list of 'columns' available in the loaded dataset.
      To get the shape tuple, access the key 'shape' (e.g., result['shape']).  To get the list of columns, access the key 'columns' (e.g., result['columns']).
    """
    global df

    # 1. Try loading the pre-merged CSV first
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    
    # 2. Fallback: Load and merge all yearly CSV files if the main path doesn't exist
    elif os.path.exists(FALLBACK_PATH):
        # Find all CSV files in the fallback directory (e.g., 2015.csv, 2016.csv, etc.)
        df = pd.DataFrame([])
        for filename in os.listdir(FALLBACK_PATH):
            if filename.endswith('.csv'):
                curr_df = pd.read_csv(f"{FALLBACK_PATH}/{filename}", delimiter=';', decimal=",")
                year = filename.split("_")[2].split(".")[0]
                curr_df['year'] = year
                if year == "2024":
                    curr_df.columns = curr_df.columns.str.replace('Ladder score', 'Happiness score')
                df = pd.concat([df, curr_df])
        df['year'] = df['year'].astype(int)
        df = df.dropna()
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(' ', '_')
            
    # 3. Guard rail: If both steps failed to create a DataFrame, return the error dict
    if df is None:
        return {
            "error": "Unable to load data. Both DATA_PATH and FALLBACK_PATH were inaccessible or empty."
        }

    # 4. Success: Return the metadata dictionary exactly as requested
    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }

@tool
def summarize_column(column: str) -> dict:
    """
    Returns descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The name of the column to analyze.

    Returns:
        A dictionary containing summary statistics (mean, max, min, etc.) or an error message if data is missing or the column is invalid.
    """
    global df
    if df is None:
        return {"error": "No data has been loaded yet. Please run load_happiness_data first."}
    
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in the dataset."}
        
    return df[column].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Computes the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The name of the first column.
        col2: The name of the second column.

    Returns:
        dict: A dictionary containing 'col1', 'col2', 'pearson_r', and 'p_value' 
              rounded to 4 decimal places, or an error message on invalid input.
    """
    global df
    if df is None:
        return {"error": "No data has been loaded yet."}
        
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both of the specified columns do not exist."}
        
    try:
        # Drop NaNs dynamically for the calculation to prevent errors
        clean_data = df[[col1, col2]].dropna()
        r_val, p_val = pearsonr(clean_data[col1], clean_data[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r_val), 4),
            "p_value": round(float(p_val), 4)
        }
    except Exception as e:
        return {"error": f"Failed to compute correlation: {e}"}

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Returns the top N countries ranked by a given metric column for a specific year.

    Args:
        column: The metric column name to rank countries by (e.g., 'Log GDP per capita').
        year: The specific calendar year to filter the data by.
        n: The number of top results to return. Defaults to 5.

    Returns:
        dict: A dictionary containing a list of record dictionary under the 'top_countries' key, where each record contains 'country' and the requested column value.  Returns an error message on bad input or empty data.
    """
    global df
    if df is None:
        return {"error": "No data has been loaded yet."}
        
    if column not in df.columns or 'year' not in df.columns or 'country' not in df.columns:
        return {"error": "Required columns ('country', 'year', or the requested metric) are missing."}
        
    try:
        # Filter by year
        year_df = df[df['year'] == year]
        if year_df.empty:
            return {"error": f"No data found for the year {year}."}
            
        # Sort and pick top N
        top_n = year_df.sort_values(by=column, ascending=False).head(n)
        
        # Format output as a list of dicts with just country and the chosen column
        results = top_n[['country', column]].to_dict(orient='records')
        return {"top_countries": results}
    except Exception as e:
        return {"error": f"Failed to retrieve top countries: {str(e)}"}


#Task 2: Build the agent


model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


if __name__ == "__main__":
    #Task 3: Run Guided Queries
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    #Task 4: Your Own Questions
    # My query 3
    my_query_3 = "Is the country with the highest log_gdp_per_capita in 2021 also among the top 5 countries for life_ladder in that same year?"
    response_3 = agent.run(my_query_3, reset=False)
    print(response_3)

    # My query 4
    my_query_4 = "What are the descriptive statistics for the social_support column in our dataset?"
    response_4 = agent.run(my_query_4, reset=False)
    print(response_4)


#Task 5: Reflection
# --- Reflection ---

# 1. In Query 3, how did the agent communicate whether the correlation was statistically significant? Did it use the p-value correctly? What threshold did it apply?
#A: It communicated successfully by outputting "statistically_significant: True" after checking the tool's results.  It used the p-value correctly by interpreted 0.0 as statistical significant at alpha threshold of 0.05.

# 2. Did any of the agent's responses surprise you — either by being more capable than you expected, or less? Describe one specific example.
#A: I was surprised about how the agent repeated the same error and handled it.  In the plot task, after many attempts, it hardcoded the dataset from scratch to create and save the required plot.

# 3. What one additional tool would make this agent meaningfully more useful? Describe what it would do and what kind of question it would help the agent answer.
#A: Currently there isn't a function tool that provides all the rows from all the years at once.  So, this tool is needed when the agent needs to plot trends over the years.
