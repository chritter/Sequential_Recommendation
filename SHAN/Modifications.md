# Modifications and Notes for STC

Christian Ritter

* SHAN by Ying18, implementation of https://github.com/chenghu17/Sequential_Recommendation and file pre-processing
adopted from https://github.com/uctoronto/SHAN/preprocess

#### Comments

* There is no data set
* no choice of hyperparameter
* Code is optimized to run the `tafeng` dataset, which has a particular structure. The data readers 
are optimized for this. Would need to adopt for our purposes.
* Pre-processing
    * do not consider users with less than 2 sessions
    * Prediction items are randomly choosen from each session and user. We do not try to predict all-1 items for each session!
    * Considers in test set only users which appeared in training set! Major limitation as it cannot be applied to new users
* Precision@k is actually Recall@k. Mix-up!    
    


### Input Data

#### A) Input for pre-processing 
as shown below, sorted by UserId and Time (Gowalla data cleaning ment removing users with 3 or less sessions,
deduplication).

UserId	ItemId	Time
0	9410	1282243748.0
0	19542	1282226349.0

Script generate_session_Gowalla then creates the input for SHAN as shown in B)

####B) Input for SHAN

* Expects data in the format where first column is `user`, second column is `sessions`. 
In the second columns we have a string containing items which are separated by @, with itemIDs between
sessions. However the first row however needs to contain the total number of users and total number of items, in the 
 
user 	sessions
123 12312312
0 	0:1@2:3@4:5:6:7:8:9:10:11@3:12@13:14:15@16:17:...
1 	24:25:26:27:28:29@30:24:31:32:33:27:34:35@32:2...

* Note that users in test set must have appeared in training set