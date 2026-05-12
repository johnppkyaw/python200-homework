from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

#Step 1: Setup
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

folder_dir = "./resources/groundwork_docs"
docs_dir = Path(folder_dir)
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

#Step 2: Load the Documents
# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader(folder_dir).load_data()
print(f'{len(docs)} documents were loaded:')
for doc in docs:
    print(doc.metadata['file_name'])

#Step 3: Build the Index and Query Engine
#Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)
#Create a query engine with similarity_top_k=3
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")

#Step 4: Query the Assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for q in questions:
    print(f"Q: {q}")
    response = query_engine.query(q)
    print(f"A: {response}")
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)

#Q: After running all five queries, add a comment reflecting on the responses: did the assistant sound confident and accurate? Did any of the answers surprise you?

#A: The assistant sounded confident and accurate; it reports what's from the source.  None of the answers surprised me except when the model was asked the dairy-free milk options question, it has higher similarity score for seasonal_specials.txt than the menu.txt.

#Step 5: Find a Failure
failure_question = "What am I wearing right now?"
print(f"Q: {failure_question}")
response = query_engine.query(failure_question)
print(f"A: {response}")

for node_with_score in response.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)

"""
    What you asked and why you expected it to be hard
    What went wrong — wrong retrieval, missing information, the model guessed anyway?
    When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?
    What you would change about the system to improve it?
"""

#A: I asked what I was wearing and I expected it to be hard as it will not be found in the documents.  The model retrieve the documents but with low similarity score.  It did not guess and admitted that it could not provide the answer due to the unrelated question to the context information.
# When the retrieval failed, the model's tone changed.  It became less confident.  AI is only as good as the data provided and there is a chance that it will hallucinate when it's not well-trained.  It's best to check the similarity score to determine whether we should trust the model's answer.  To prevent the model from answer the unrelated question, I would add a threshold where the model should answer that it does not know the answer when the similarity score is less tha 0.80.  I would also add an instruction to only answer the question related to the Groundwork coffee shop.


#Step 6: Reflection
#Q: The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?

#A: It only took 2 lines in my project.  Using a framework helps manage all the complex steps reducing the low level codes and helping us focus on the logic of the application.


#Q: You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.

#A: A tutoring assistant where the model will answer all the user questions only relevant to a textbook of choice.

#Q: What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

#A: RAG may have issue in retrieving when more than one document have contradicting information on the same topic (store hours, for example).  Unless a specific instruction was given, the model will have no way of knowing which info is up to date and true.  
