# Part 1: Portal Walkthrough

Video link: https://www.youtube.com/watch?v=61dH0J_ofVM

# Part 2: Cost Analysis and Part 3: Python in Cloud Shell

Per the Azure Pricing Calculator, lightweight compute scenario, running 160 hours a month, costs $1.66 a month and the heavy analytics workload scenario, running 24/7, costs $2997.30 a month.  These numbers don't surprise me as it will definitely cost a lot more to run a bigger machine at 24/7 than a smaller machine at limited hours.  

One surprising thing was that the cost rate is about the same between Azure Machine Learning and the VM for the same instance of NC6s V3.  Depending on the user preference, they can either start everything from scratch with the VM or everything setup with the Azure Machine Learning to start training the model.

The script printed that "Scenario B VM costs 1396.1x more than Scenario A" and the calculated costs matched with what I saw in the Pricing Calculator.  In Scenario B, it did instruct to also add the cost of the database and the storage account but in the script, it only compares just the cost of running the machines, so that's where the discrepancy exists when we don't remove the cost of those two services and compare the result with the script's.
