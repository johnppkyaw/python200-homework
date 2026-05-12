from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---

#Concepts Question 1
"""
Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.

--I recommend RAG because of the frequent updates to the policy.  With this method, the model will have the direct access to all the changes and will effectively provide the specific section of the documents based on the query.

Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.

--I recommend fine-tuning the model using the 3000 samples.  This will ensure that the model will be consistent in writing in their desired brand voice without prompt engineering.

Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.

---For this case, I recommend prompt engineering. Since the work needs to be done only one time, it will take a lot more time to set up a RAG pipeline or fine-tuning than actually finishing the task via prompt engineering.
"""

#Concepts Question 2
"""
Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.
Think about the tone of the response as well as its content — why does the way the model expresses an answer affect how much we trust it?

---A confidently wrong answer is more harmful because the user will take the wrong information without having any doubts.  For example, taking a bad medical advice from the model without knowing it's confidence can be deadly to the user.  Because the model expresses confidently in complete sentences in providing both true and false statements, there is no way to know if it's true or not.

"""


#Concepts Question 3
"""
steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query"
]


arranged_steps = [
    1) "Extract text from source documents",    
    2) "Split text into chunks",
    3) "Convert text chunks into embeddings",
    4) "Receive the user's query",
    5) "Embed the user's query",
    6) "Retrieve the most relevant chunks",
    7) "Inject retrieved chunks into the prompt",
    8) "Generate a response from the LLM"
]

    1) Data is extracted from the document and converted into the readable text format
    2) The text is broken into smaller chucks that are easily processable by the model
    3) These chunks are then converted into vectors.
    4) System receives the question or instruction provided by the user to start the search process
    5) The user's input is converted into the vector by the same model to compare the meanings.
    6) The system searches in the database the most similar embedding to the embedding of the user's query 
    7) The most relevant segment is then sent to the prompt to provide the model with the facts it needs to respond.
    8) Model uses the provided context and provide the answer for the user.

"""

#Keyword RAG
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


#Keyword Question 1
query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

simple_keyword_retrieval(query, documents)

#Q: After running the function, add a comment explaining which document was selected and why.

#A: loyalty.txt was selected because only hiring.txt and loyalty.txt have the same number of overlap 1 but after the list was sorted in descending order, loyalty.txt was the first on the list.  Also, the model could not tell the similarity between weekend and weekends or the hours context.

#Keyword Question 2
query = "Do you have anything without caffeine?"
simple_keyword_retrieval(query, documents)

"""
Which document was selected
Whether keyword RAG got this right — and why or why not
What kind of retrieval would do better here
"""

#A: No document was selected.  Keyword RAG got this wrong because it relies on literal string matching and missed Oat milk and almond milk from menu.txt.  Semantic Search would do better here as it understands the relationship and meanings between the words.

#Keyword Question 3
query = "How do I sign up for rewards?"

#Q: Before running any code, predict which document will be selected for the query below. Write your prediction and your reasoning as a comment first, then run the code to check.

#A: No document will be selected because there is no words matching.

simple_keyword_retrieval(query, documents)

#Q: Was your prediction correct? If the result surprised you, add a comment explaining what happened.

#A: My prediction was correct.  

#--- Semantic RAG Concepts ---
#Semantic Question 1

#Q: What is a vector embedding? (1-2 sentences)
#A: Vector embedding is when a numeric data is converted into a list of numbers called tensor.  This allows the computer to calculate the relationship among other data.

#Q: Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
#A: The chunk with the cosine similarity score of 0.85 is more relevant with the given query as it's close to 1.  The score of 0.30 means that the chunk is largely unrelated to the user's query.


#Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
#A: It represents the text as a vector and group the similar and related text together.  So, during the search it will look for not just the exact word but the words with similar meaning as the query.

#Semantic Question 2
"""
| Feature                    | Keyword RAG                       | Semantic RAG        |
|----------------------------|-----------------------------------|---------------------|
| What is compared?          | Exact word overlap                | Vector embeddings   |
| What is retrieved?         | Full document                     | Text chunks         |
| Can it handle synonyms?    | No                                | Yes                 |
| Storage format             | Plain text dictionary             | Vector database     |
| Relevance score            | Number of overlapping keywords    | Similarity score    |
"""

#LlamaIndex
#LlamaIndex Question 1
import os
from pypdf import PdfReader
from llama_index.core import Document, VectorStoreIndex

# Load documents directly from PDFs in the folder
def get_documents_from_folder(folder_path):
    documents = []
    
    # Loop through every file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Extracting text from: {filename}")
            
            try:
                reader = PdfReader(file_path)
                extracted_text = ""
                
                # Extract text page by page
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
                
                # Only create a document if text was actually found
                if extracted_text.strip():
                    doc = Document(
                        text=extracted_text, 
                        metadata={"file_name": filename}
                    )
                    documents.append(doc)
                else:
                    print(f"Warning: No text found in {filename} (might be a scan/image).")
                    
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                
    return documents

# 1. Load the clean documents
folder_dir = "./resources/brightleaf_pdfs" # Change to your folder path
docs = get_documents_from_folder(folder_dir)

# 2. Build your index with the clean text
index = VectorStoreIndex.from_documents(docs)

print(type(index._vector_store).__name__)
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

#Q: Do the retrieved chunks look relevant to the question?
#A: The retrieved chunks look relevant to the question.

#Q: Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
#A: The model sounds confident and specific.

#Q: Did anything unexpected get retrieved?
#A: In the security query, "Employee Well-being" appeared as a top result but even with the wrong document retrieved, the model was able to extract the correct text to generate the response.

#LlamaIndex Question 2
#similarity_top_k=1
print("similarity_top_k=1")
query_engine = index.as_query_engine(similarity_top_k=1)
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

#similarity_top_k=5
print("similarity_top_k=5")
query_engine = index.as_query_engine(similarity_top_k=5)
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

#Add a comment explaining how the response changed (if at all) and whether more retrieved context is always better.
#A: At k=1, the response is a concise summary but lacking details.  At k=5, the model responds with a lot more detail and is noted to have unexpected retrievals, such as reading EcoVolt Energy Partnership and Financial Performance document for the security question.  Therefore, more retrieved context is not always better.  

#LlamaIndex Question 3
query = "What is the company's policy on bringing my lamborghini to the office?"

print(f"\nQ: {query}")
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(query)

print("A:", response)

for node_with_score in response.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)

#Q: Add a comment explaining what you expected, what actually happened, and what you would change about the system to handle this kind of query better.

#A: I expected the model to say it couldn't find any information about lamborghini, since that is not available in the PDFs.

# What actually happened was it still retrieved the chunks even though none of them mentioned about my lamborghini and it did mention that there were no related info.

# I would add a "similarity threshold" to retrieve the chucks only if the similarity score is > 0.7.  Otherwise, if there are no match > 0.7, the model should respond with just "I don't know."


#LlamaIndex Question 4
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# Get response to query
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)
print("Evaluation for the first query: ")

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

print("Evaluation for the second query: ")
# Get response to query
q = "What employee benefits does CodeTheDream offer?"
response = query_engine.query(q)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

"""
What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
What does a relevancy score measure, and how is it different from faithfulness?
Did the scores change between your two queries? If so, why do you think that happened?
What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?

A: A faithful score of 1.0 means every claim in the response can be referred back to the document snippets and a score of 0 means the response is not found in the retrieved chunks.

Relevance score measures how useful the response is to the user's question.  It's different from faithfulness by measuring if the available information is enough to address the user's question.

The score change between the two queries because there is no relevant information for the 2nd query in the documents.

In LLM-as-a-judge approach, the metrics are computed by an external LLM.  The LLM will be able to assess the subjective metrics faster compared to multiple human experts. 
"""
