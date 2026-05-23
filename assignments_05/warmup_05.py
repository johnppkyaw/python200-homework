from dotenv import load_dotenv
from openai import OpenAI

#The Chat Completions API
#API Question 1
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
print(f'The response text: {response.choices[0].message.content}')
print(f'The name of the model that responded: {response.model}')
print(f'The number of tokens used: {response.usage.prompt_tokens}')

#API Question 2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for each_temp in temperatures:
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": prompt}],
    temperature=each_temp
  )
  print(f'The response text with temp of {each_temp}: {response.choices[0].message.content}')

#Q: What do you notice about how the outputs differ? Which temperature would you use if you needed a consistent, reproducible output?
#A: The outputs differ on each run with temp 0.7 and 1.5.  It gives a different name each run and also add a reasoning to the name. The output is the same with the temp of 0.  Therefore, I would use the temp of 0 for a consistent, reproducible output.

#API Question 3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for index, each_response in enumerate(response.choices):
  print(f"Response #{index}: {each_response.message.content}")

#API Question 4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
    temperature=0
)
print(f'The response text with max tokens of 15: {response.choices[0].message.content}')

#Q: What happened, and why might you want to use max_tokens in a real application?
#A: The response was limited.  Setting the max_tokens will help keep the costs under control in a real application.

#System Messages and Personas
#System Question 1

messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0
)
print(f'The response text to list comprehension 1: {response.choices[0].message.content}')

messages = [
    {"role": "system", "content": "You are a very angry but kind tutor.  You explains things in a short and concise way."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0
)
print(f'The response text to list comprehension 2: {response.choices[0].message.content}')

#Q:Now change the system message to give the model a completely different personality (your choice) and ask the same question. Print that response too. Add a comment noting what changed.

#A:The personality changes based on the context added in system content.

#System Question 2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0
)
print(f'The response text (System Question 2): {response.choices[0].message.content}')

#Q: Why does the model know Jordan's name, even though it's stateless?
#A: The model does not remember Jordan's name after each interaction since it's stateless.  It knows only because Jordan's name from the chat history was included in the messages array with each interaction.

#Prompt Engineering

#Prompt Question 1 — Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

prompt = "Classify the sentiment of each review below as positive, negative, or mixed: "
for index, review in enumerate(reviews):
   prompt_str = prompt + review
   print(f'Review #{index}: {get_completion(prompt_str)}')


#Prompt Question 2 — One-Shot
prompt = """Classify the sentiment of each review below as positive, negative, or mixed. 

Example:
Review: 'Fast shipping but the item arrived damaged.'
Sentiment: mixed

"""
for index, review in enumerate(reviews):
   prompt_str = prompt + review
   print(f'Review #{index}: {get_completion(prompt_str)}')

#Q: Did adding one example change the format or consistency of the output compared to Q1?
#A: It changed the format of the output compared Q1.  Instead of providing full sentence, it just answers with ."Sentiment: "

#Prompt Question 3 — Few-Shot
prompt = """Classify the sentiment of each review below as positive, negative, or mixed. 

Example:
Review: 'Fast shipping but the item arrived damaged.'
Sentiment: mixed

Review: 'Excellent performance and a very user-friendly design. It works exactly as advertised!'
Sentiment: positive

Review: 'Poor build quality and frustrating customer service. I wouldn't recommend this to anyone.'
Sentiment: negative

"""
for index, review in enumerate(reviews):
   prompt_str = prompt + review
   print(f'Review #{index}: {get_completion(prompt_str)}')

#Q: Add a comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?

#A: 
#zero-shot: choose this for simple tasks and the model has pre-trained knowledge and also to minimize token usage.
#one-shot: chose this when specific output format is needed in the response
#few-shot: choose this when a complex reasoning is needed at high accuracy.

#Prompt Question 4 — Chain of Thought
problem = "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later takes a new job that pays $7,500 more per year than her post-raise salary. What is her final annual salary?"

prompt = f"""solve the following problem, but instruct it to show its reasoning step by step before giving a final answer: 

{problem}
"""

print(f'Prompt Q4 Response: {get_completion(prompt)}')
#Q: Why does asking the model to reason step by step tend to improve accuracy on problems like this?
#A: Asking the model to reason step by step makes sure the model actually follows the math through each step, instead of just taking a "lucky guess" at the output.

#Prompt Question 5 — Structured Output
import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""analyze the review below and return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1), and reason (one sentence).: 

{review}
"""

raw_response = get_completion(prompt)
print(f'The raw response: {raw_response}')

try:
   data = json.loads(raw_response)
   for each_field in data:
      print (f'{each_field}: {data[each_field]}')

except json.JSONDecodeError as e:
   print("Invalid JSON syntax:", e)
   print("raw response: \n", raw_response)
   

#Prompt Question 6 — Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
print(f'Prompt Q6 Response with instruction: {get_completion(prompt)}')

user_text_no_instructions = "Stout is a type of a beer.  Hamburger is a type of food."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text_no_instructions}```
"""

print(f'Prompt Q6 Response without instruction: {get_completion(prompt)}')

#Q: What problem do delimiters help prevent?
#A: They prevent the model from confusing the instructions, the examples, and the actual user's input. This will also make sure that the user cannot bypass the restrictions included in the instruction.

#Local Models with Ollama

#Ollama Question 1

#Ollama output
"""
A large language model is an AI system trained on vast amounts of text to understand and generate human language, 
enabling it to grasp context, nuances, and complex ideas. It processes massive datasets to learn patterns and 
improve its ability to create meaningful text, making it a powerful tool for tasks like writing or answering 
questions.
"""
prompt = "Explain what a large language model is in two sentences."
print(f"OpenAI's response: {get_completion(prompt)}")

#Q: What differences did you notice between the two responses? 
#A: OpenAI's response has more technical terms and Ollama's response focuses more on the context and meaning. 

#Q: What is one advantage and one disadvantage of running a model locally?
#A: An advantage of running a model locally is privacy; my data are safe in the local machine.  An disadvantage is its performance limited by the local computer's GPU and RAM power.
