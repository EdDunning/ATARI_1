# ATARI-1
Automated Surgical Skill Assessment in Robotic Cardiac Procedures: A Motion Analysis and ML Framework

Task 1. 
Define and extract motion features from robotic procedure data: smoothness, economy of motion, path length, and jerk, grounded in a control-theory and biomechanics framing.

motion_feature_extraction.py contains the functions to calculate these metrics for each file

motion_feature_storage.py runs all the files through the functions in motion_feature_extraction.py and compiles the data into three csv files, one for each dataset.

Task 2. 
Map motion features to expert vs. novice skill ratings using an OSATS-style reference standard.

Statistical analysis to find the correlation coefficients between both the motion features and the OSATS as well as the OSATS and the experience level.

This is more of a stats study of the data as the outputs of this are not used for the classifiers. Its results are nontheless informative.

Task 3. 
Train and validate an ML classifier for objective skill assessment, reporting agreement with expert ratings.

Classifier 1: RandomForest, pooled tasks
Classifier 2: RandomForest, task specific
Classifier 3: XGBoost, task specific
Classifier 4: Linear regression: non-task specific

Task 4. 
Frame outputs for training feedback and credentialing applications. Coordinate task split with Jack Hemphill and the senior team.

GUI: I made a web based Graphical User Interface for a user to navigate and run a kinematic data file to extract all motion features, as well as run Classifier 2 with its outputs. This will be for an intended use of inputting a file which is not part of the dataset that the model has been trained on.

