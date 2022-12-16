# Amazon Food Review Evaluation

Here are the steps for AmazonFoodReview.py code: 

* Step 1: read data from "Reviews.csv" and store it in dataframe df.
* Step 2: create a new column "Sentiment" for positive / negative, and keep only columns "Score", "Sentiment", "Summary", "Text". 
* Step 3: clean the data using function cleanup(). 
* Step 4: split data into train and test.
* Step 5: convert text data into numeric vectors, separately for uni_gram, bi_gram, and tfidf. 
* Step 5: train models (logreg_uni_gram, logreg_bi_gram, logreg_tfidf, rf_bi_gram) and store the evaluation in prediction and prob for visualization in notebook report. 
