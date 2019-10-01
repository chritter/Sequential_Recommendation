# Modifications and Notes for STC

Christian Ritter

* SHAN by Ying18, implementation of https://github.com/chenghu17/Sequential_Recommendation and file pre-processing
adopted from https://github.com/uctoronto/SHAN/preprocess

#### Comments

*  No choice to set hyperparameter as input, except embedding dimensions. learning rate could be set through code modification. 
* Code is optimized to run the `tafeng` dataset, which has a particular structure. The data readers 
are optimized for this. Would need to adopt for our purposes.
* Pre-processing
    * do not consider users with less than 2 sessions
    * Prediction items are randomly choosen from each session and user. We do not try to predict all-1 items for each session!
    * ***Considers in test set only users which appeared in training set! Major limitation as it cannot be applied to new users***
* Precision@k is actually Recall@k. Mix-up!    
    


### Input Data

#### I) Data pre-processing by the authors

#### Ia)

Pre-processing with clean_data_Gowalla.py on data of the form

0	2010-10-18T22:17:43Z	30.2691029532	-97.7493953705	420315
0	2010-10-17T23:42:03Z	30.2557309927	-97.7633857727	316637

Filter and clean data:
* removed duplicates
* sort by user_ID and time
* retain only users with more than 3 sessions
* write out columns: ['use_ID', 'ite_ID', 'time'] in the form:

UserId	ItemId	Time
0	9410	1282243748.0
0	19542	1282226349.0

#### Ib)

Pre-processing with generate_session_Gowalla.py. 

which needs to be sorted by UserId and Time (Gowalla data cleaning ment removing users with 3 or less sessions,
deduplication).

* Train and test splitting: Note that users in test set must have appeared in training set
* Formatting: This creates the output as input for SHAN in the format where first column is `user`, second column is `sessions`. 
In the second columns we have a string containing items which are separated by @, with itemIDs between
sessions. However the first row however needs to contain the total number of users and total number of items, in the 
 
user 	sessions
123 12312312
0 	0:1@2:3@4:5:6:7:8:9:10:11@3:12@13:14:15@16:17:...
1 	24:25:26:27:28:29@30:24:31:32:33:27:34:35@32:2...

This is the input for SHAN.

### II) Training and Evaluation

#### Ia)
Execute shan.py to run  the model, evaluation takes place after each iteration/epoch and will be printed out.

#### Ib)
There is no separate function for predictions! 



#### I) Data pre-processing by STC

* Make sure data is sorted as in step Ia). We need to retain only users with 2+ sesssions?
In fact we need at least 2 sessions, as 1 has to be in the training set and 1 in the test set! Authors retain 3, why?
* Use same train-test split as for GRU4Rec, need to adapt the Gru4Rec splitting. 


#### I) Performance testing by STC
* Requires creation of new train,test set which is a subset of the Gru4Rec train,test sets! Need to evaluate on those
with other models for fair comparison


