# Speech command recognition for drone instructions
This is an academic project for the Audio Pattern Recognition course in the Master Degree in Computer Science at the Università degli Studi di Milano
In this repo you can find a report containing a more in depth explanation of the work done.
All the code is made in MATLAB
## Contents of the repo
- A report of the work done
- *BuildDataset.m* a script to install the database
- *augmentDataset.m* a function developed by MATLAB to add the background samples to the dataset
- *trainNet.m* a script to train a net to recognise the commands
- *realTimeTest* a script to classify commands given at realtime using the trained net
- *trainedNet.mat* and *classes.mat* a pre-trained net with its labels
