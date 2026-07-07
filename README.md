# ATARI-1
Automated Surgical Skill Assessment in Robotic Cardiac Procedures: A Motion Analysis and ML Framework

•	Define and extract motion features from robotic procedure data: smoothness, economy of motion, path length, and jerk, grounded in a control-theory and biomechanics framing.
motion_feature_extraction.py does this. 
It is currently at the stage where it can do this for one kinematic data file. The next step is to write another file to store these outputs for all files. 
motion_feature_storage.py will do this.

•	Map motion features to expert vs. novice skill ratings using an OSATS-style reference standard.

•	Train and validate an ML classifier for objective skill assessment, reporting agreement with expert ratings.

•	Frame outputs for training feedback and credentialing applications. Coordinate task split with Jack Hemphill and the senior team.

