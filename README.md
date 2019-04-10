# Naive-Bayes-for-Text-Classification
Chinese news classification by naive bayes

# Algorithm Theory
This item doing Chinese news classification by naive bayes,so the kernel of naive bayes is bayes formula,like below:
![image](https://github.com/FelixHuangX/Naive-Bayes-for-Text-Classification/blob/master/%E5%85%AC%E5%BC%8F.JPG)

When we get an feature vector X,the task of this algorithm is estimate P(Y|X) based on bayes formula.And P(X) is known because of we already get feature vector X.So the mask task of this model training is parameter estimating for P(X|Y) and P(Y).

# Data
The data folder named 'cnews' means Chinese news,which include 4 files:'cnews.test.txt','cnews.train.txt','cnews.val.txt','cnews.vocab.txt'.The 'cnews.train.txt' file is very big,which obtian 50k news with 10 calsses,so you can try 'cnews.val.txt' frist,and 'cnews.vocab.txt' is a stopwords list for remove stopwords while participling.And detail of the data like below:


# WordCloud
The code ''
