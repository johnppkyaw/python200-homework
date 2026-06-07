#--- Azure Authentication ---
#Azure Authentication Question 1
#Q: when you run a Python script locally that uses DefaultAzureCredential, what does it rely on to authenticate? What command must you have run first, and how does DefaultAzureCredential know to use it?

#A: It relies on the sequence of authentication methods such as environmental variables, managed identity and az login section and stops at the first one that works.  az login must be run first.   DefaultAzureCredential knows to use it via its checklist and can pick up any active session automatically. 

#Azure Authentication Question 2
#Q: why can't a deployed pipeline (running on an Azure VM or container) use az login for authentication? What does it use instead, and why does the same Python code work without changes?

#A: Because there is no human around to run the az login command.  It uses managed identity assigned to the VM or the container and the code works without any changes because the DefaultAzureCredential checks the chain of the authentication methods until it finds the first authentication method that works.

#Azure Authentication Question 3
#Q: You run a script that creates a DefaultAzureCredential and immediately gets an AuthenticationError. In a comment block, describe the two most likely causes and how you would diagnose each.

#A: 1) az login session has expired, which can be resolved by running the az login command again. 
#2) Managed Identity is not active which can be resolved by checking if there is an active resource group and subscription.

#--- Blob Storage ---

#Blob Storage Question 1
#Q: describe the three-level hierarchy of Azure Blob Storage in your own words. Give a concrete analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.).

#A: A storage account is the top-level resource like a filing cabinet.  Each account has a unique name in Azure and a URL of the form.  A container a grouping of blobs within the storage account like a folder in that filing cabinet.  A blob is an individual file like a document in the folder.

#Blob Storage Question 2
#Q: For each scenario below, write one sentence in a comment block saying whether you would use Blob Storage or a relational database (like Azure SQL), and why.

#A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
#Answer: I would use Blob Storage since we are just storing and retrieving them as-is for future processing.

#Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
#Answer: I would use Azure SQL as it is more efficient in querying the data than the Blob storage.

#A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.
#Answer: I would use Blob Storage since we are just storing the entire dataset rather than querying.

#Blob Storage Question 3
#Q: Write a function list_container(container_client) that prints the name and size (in bytes) of every blob in the container, one per line. The function should take a ContainerClient object as its only argument and return nothing.
"""
def list_container(container_client):
  for blob in container_client.list_blobs():
    print(f"  {blob.name}  ({blob.size} bytes)")
"""

#Blob Storage Question 4
#Write a function upload_text(container_client, blob_name, text) that encodes a Python string as UTF-8 and uploads it as a blob, overwriting any existing blob with the same name. The function should take a ContainerClient, a blob name string, and a text string, and return nothing.

"""
def upload_text(container_client, blob_name, text):
  payload = text.encode("utf-8")
  container_client.upload_blob(blob_name, payload, overwrite=True)
"""
