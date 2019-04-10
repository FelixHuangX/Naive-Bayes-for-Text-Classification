# Naive-Bayes-for-Text-Classification
Chinese news classification by naive bayes

# Algorithm Theory
This item doing Chinese news classification by naive bayes,so the kernel of naive bayes is bayes formula,like below:
![image](https://github.com/FelixHuangX/Naive-Bayes-for-Text-Classification/blob/master/%E5%85%AC%E5%BC%8F.JPG)

When we get an feature vector X,the task of this algorithm is estimate P(Y|X) based on bayes formula.And P(X) is known because of we already get feature vector X.So the mask task of this model training is parameter estimating for P(X|Y) and P(Y).

# Data
The data folder named 'cnews' means Chinese news,which include 3 files:
'cnews.test.txt','cnews.train.txt'(.zip),'cnews.vocab.txt'.
The 'cnews.train.txt' file is very big,which obtian 10k news with 10 calsses,and 'cnews.vocab.txt' is a stopwords list for remove stopwords while participling.And detail of the data like below:
![image](https://github.com/FelixHuangX/Naive-Bayes-for-Text-Classification/blob/master/%E6%95%B0%E6%8D%AE%E7%A4%BA%E4%BE%8B.JPG)

# WordCloud
The code 'Naivebayes_train.py' include the Wordclou part,and this section has been covered in this .py file.For example,Wordcloud can show the mian words in each class of Chinese news,like below:

The Sport News:
![image](https://github.com/FelixHuangX/Naive-Bayes-for-Text-Classification/blob/master/wordcloud/0.JPG)

The Current Politics News:
![image]( https://github.com/FelixHuangX/Naive-Bayes-for-Text-Classification/blob/master/wordcloud/6.JPG)

The Finance News:
![image](https://github.com/FelixHuangX/Naive-Bayes-for-Text-Classification/blob/master/wordcloud/9.JPG)

# Execution
First run:`Naivebayes_train.py`,and the model will be saved in the folder 'models'.
Then run:`NaiveBayes_test.py` to test accuracy in test data.I already done this test,and the accuracy can be up to 95%.
![image]()
