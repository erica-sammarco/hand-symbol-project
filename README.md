## File Organization

### Calibration Program
calibration.py - Runs the calibration program, writes results to txt files in the ./results folder  
constants.json - Sets the parameters for the calibration program  
./poses - Any .jpeg or .png file directly in the ./poses folder will be used for the calibration program  

### Data Processing
graph-results-pipeline.ipynb - Reads a result txt file, parses the data, plugs it into the normalization and graphing ("high five") process 

### Complete Experiment
./experiment - contains the stages of the experiment from calibration to keyboard