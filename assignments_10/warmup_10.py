# --- LLMs as Transform ---

# LLMs as Transform Question 1

#For each task below, write a one-sentence comment saying whether you would use an LLM or deterministic code, and why.

"""
--Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
  I would use deterministic code because a function can easily convert the date format.

--Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
  I would use LLM because the task requires judgment and classification.

--Calculate the average of a list of numbers. 
  I would use deterministic code because a math function can easily calculate it and it's 100% accurate.

--Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)".
  I would use LLM because the task requires field extraction and LLM can detect anywhere in a text.

--Determine whether a product review is more than 100 words long.
  This task can be easily accomplished by writing a function that can count the number words in a string.  I don't think an LLM is needed here.
"""

# LLMs as Transform Question 2


#Your colleague has written the following pipeline prompt:

#system = "Summarize this product review in a few sentences."

#Q: In a comment block, explain what problem this creates downstream in a pipeline, and rewrite the prompt so it produces output that is easy to parse and store reliably.
#The prompt may create a review useful for a human reader but will be very difficult to parse in a pipeline.  It needs to constrain the output to a known set of values.  It can be rewritten like this: "Classify this product review.  Replay with exactly one word: positive, negative or neutral."


#LLMs as Transform Question 3

#Your dataset has 50,000 records and you need to run a classification call for each one using gpt-4o-mini. In a comment block, answer:

#Q:  If each call takes 1 second on average, how long would sequential processing take? 
#A: 50,000 records would take 50,000 seconds which takes about 13.8 hours to process

#Q: What is one practical strategy to handle this more efficiently at scale, without changing models?
#A: Use OpenAI's Batch API for the asynchronously processing at a reduced cost.

# --- Azure OpenAI ---

# Azure OpenAI Question 1
#Q: In a comment block, name two reasons an organization might use Azure OpenAI instead of calling the OpenAI API directly. Be specific -- "it's better" is not an answer.

#A: The data does not leave the from Azure's infrastructure which is important for companies with strict data governance policies.  Azure also prohibits using the API data for training, under the Microsoft's enterprise agreement.

# Azure OpenAI Question 2
#Q: When you switch from OpenAI to AzureOpenAI, the client initialization takes three Azure-specific parameters. In a comment block, name them and describe what each one is. (Do not include the standard api_key -- describe the Azure-specific ones.)

#A: azure_endpoint: the base URL of the deployed Azure OpenAI resource,
#api_version: the specific version of Azure OpenAI REST API
#azure_deployment: the custom name of the specific model deployment inside Azure AI studio


# Azure OpenAI Question 3
#Q: In a comment block, answer: when using AzureOpenAI, the model parameter in chat.completions.create() does not take a value like "gpt-4o-mini". What does it take instead, and where do you find the right value to use?

#A: The model parameter takes a deployment name, instead of a model name.  This is a named deployment created by the employer and can be found in the Azure OpenAI resource in AI Foundry.
