# Code and data for replication issue lifetime prediction in GitHub projects

Note that the results are not exactly as reported to paper, due to random seed not being fixed in paper and bug in data partitioning which left out some train data.

Requires python27 and dependencies specified in `requirements.txt`. Tested on Windows and Linux/

To run the code
 * Clone the repository, change working directory into this directory
 
 * Downlaoad the datasets from here https://www.dropbox.com/s/sjyu14jvbmkqbz4/issue_data.tar.gz?dl=0. And extra so that all data files are under folder `issue_data` directly. 
 * Create an virtualenv (make sure you are using python2)
  ```
  virtualenv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
 
 * To run the prediction and evaluation, use `pythun run_experiments.py` It requires about 8GB of RAM and takes approximately 2 hours to complete on modern laptop.
 * `results-all.csv` file contains all the performance measures, `results_fixedtest.csv` Contains the table where the test set size is fixed for different prediction tasks. 

 * Resulting feature importance figures are in `figures` folder.  
