# DSCI-644 Project Repository

**This repository consists of the items below:**
1. Raw data file - "project1-commitsRefactoring.xlsx" - Consists the base raw data excluding the duplicates (entire row occuring more than once)
2. data_train.csv - Train data extracted from Raw data file
3. data_test.csv - Test data extracted from Raw data file
4. dataone_train.csv - Train data extracted from data_train.csv consisting only single refactoring labels
5. datamulti_train.csv - Train data extracted from data_train.csv consisting only multi refactoring label
6. x_data.csv, y_data.csv, pred_data.csv - files used to analyse the fp and fn cases to understand what is going wrong here

**Note: This repository contains several experimentation codes. For Phase-3 of the project, refer to prod.py and test.py**

Steps for Execution:
1. Run test.py -> This executes prod.py file which contains the implementation. Currently this code executes dataone_train.csv (single class data)
2. To run multi-class - comment out line #105 and uncomment out line #108 to execute datamulti_train.csv (multi class data)




