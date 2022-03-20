# Context Tree Based Online Nonlinear Neyman-Pearson Classification
This is the repository for Context Tree based, Online Neyman Pearson (Tree-OLNP) Classifier described in [1]: 

This implementation also contains cross-validation of the hyper-parameters. Best set of hyper-parameters are selected based on NP-score with grid search.<br/>

# Evaluating and Using the results
Running the model will generate 6 different graphs.<br/>
These graphs correspond to transient behaviour of the model during training.<br/>
In order to look at the final results, use the latest element of each array for the corresponding metric.<br/>
Graphs of the 6 different arrays are shown below.<br/>
<img src="figures/code_output.png"><br/>

Top and bottom figures are related to train and test, respectively. The number of samples in training is related to the augmentation (explained in model parameters).<br/>
In current case, the number of training samples is ~150k. Similarly, for test figures, there are 100 data points, where each point is an individual test of the existing<br/>
Model also calculates the weights for each class in order to satisfy target false alarm declared by the user.<br/>
model at different stages of the training. Please refer to the paper for more detailed explanation.<br/>

# Running the Model with a new data set
In order to run the full pipeline with a new dataset
* Make sure downloaded data has the same fields with ./data/banana.mat
* Make sure the downloaded data is located under the data folder.
* Update the pipeline parameter showing the directory for the input data
* Include additional hyperparameters for better performance

# Expected Decision Boundaries
When input data is 2D, it is possible to visualize decision boundaries. I included 2 decision boundaries for target false alarms 0.05 and 0.2.<br/>
We use tree depth of 8.<br/>
<img src="figures/db_005.png">
<img src="figures/db_020.png">

# Importance for piecewise classifiers

Proposed context tree framework divides the space in to regions. In each region we train different classifier. 
It is possible to define the whole space by different combinations of regions. 
Each combination can be considered as seperate piece-wise NP classifier. Proposed tree framework sequantially learns corresponding weights of these classifiers.
More detailed information about context tree partitioning in NP framework can be found in [1].<br/>
We also show example regions and their corresponding partitions (piece-wise NP classifiers) in the figure below.
<img src="figures/tree_partition.png">

We also share how weights of different partitions changes as context tree processes more data. In the below figure, you can see that preference classifier is shifted from root to leaf nodes as number of sample increases.
<img src="figures/tree_weights.png">

Thanks!
Basarbatu Can

# References
[1] Can, Başarbatu, and Hüseyin Özkan. "Neyman-Pearson Classification Via Context Trees." 2020 28th Signal Processing and Communications Applications Conference (SIU). IEEE, 2020. <br/>