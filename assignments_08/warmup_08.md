# Part 1: Warmup -- Check for Understanding

## Cloud Concepts Question 1
What is the core economic model of cloud computing, and how does it differ from owning your own servers?

Cloud computing's core economic model is a pay-as-you-go system where you only pay for the resources that you use.  It differs from owning the servers by requiring a large upfront investment to buy and maintain them.

## Cloud Concepts Question 2
Question: What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each

Answer: Vertical scaling is upgrading the machine itself (CPU, RAM or GPU) when horizontal scaling is adding more machine and splitting the work across them.

vertical scaling example: Upgrading the GPU required to train a model for one afternoon.
horizontal scaling example: e-commerce website needing additional servers to handle the traffic spike during holiday shopping.

Question: For the three scenarios below, write one sentence saying which type of scaling applies and why.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
--Horizontal scaling applies since you will need more computing power temporarily.

A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
--Vertical scaling since you only need 1 machine with better GPU and RAM.

A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
--Horizontal scaling applies since the work can be split among multiple machines.

## Cloud Concepts Question 3
Before writing your definitions, classify each item in the list below as IaaS, PaaS, or SaaS. One sentence of reasoning is enough for each.

Gmail - SaaS
Azure Virtual Machines - IaaS
Azure App Service - PaaS
AWS S3 (Simple Storage Service) - SaaS
GitHub Codespaces - PaaS
Snowflake - SaaS

Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.


IaaS provides raw computing resources only and I, as a developer, am responsible in managing everything from the operating system up, including installing software, configuring security patches, and setting up the environment.  Example, Azure Virtual machines

PaaS provides infrastructure and I, as a developer, am responsible for managing my own code, deploying the app; the platform handles the rest.  Example, Azure App Service

SaaS is where you use the application that someone else builds and maintains, I as the developer is responsible for nothing but logging in and using the app. (example, Gmail)


## Cloud Concepts Question 4
What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?

A platform layer built on top of the cloud provider.  It is optimized specifically for data and analytics workloads.  We will gain speed and simplicity but will be giving up flexibility and lower cost.

## Cloud Concepts Question 5
The lesson names two situations where the cloud is probably not the right choice. What are they?
1) Setting up an initial prototype - when the dataset is small and compute demands are low
2) An immediate support is needed - It isn't as easy or fast as looking up a standard Python package

## Azure Basics Question 1
What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?

An Azure subscription a billing account that owns all the resources within it.  My resource group is mine alone and is the one that CTD shares.


## Azure Basics Question 2
Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?

ephemeral means every time the shell is closed, all the files and directories created will be deleted.  To make them persistent, the course have connected the Cloud Shell to a file share, a named storage folder in Azure.

## Azure Basics Question 3
What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?

SSH uses a key pair to authenticate the user without a password.  The private key stays on the machine and the public key is uploaded to the remote systems that I want to access.  It's safe this way because the private key is not uploaded and no one can access my files if the public key does not match the private key.

## Azure Basics Question 4
Run the following command in Cloud Shell without the --output table flag:
az account show
Paste the output into your answer. Then describe in one sentence what changes when you add --output table.

{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#jjpyae@gmail.com",
    "type": "user"
  }
}

The --output table flag formats the raw JSON format into a readable table.
