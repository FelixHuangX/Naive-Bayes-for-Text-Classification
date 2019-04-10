# Naive-Bayes-for-Text-Classification
Chinese news classification by naive bayes

# Algorithm Theory
This item doing Chinese news classification by naive bayes,so the kernel of naive bayes is bayes formula,like below:
![image](https://github.com/FelixHuangX/Naive-Bayes-for-Text-Classification/blob/master/%E5%85%AC%E5%BC%8F.JPG)

When we get an feature vector X,the task of this algorithm is estimate P(Y|X) based on bayes formula.And P(X) is known because of we already get feature vector X.So the mask task of this model training is parameter estimating for P(X|Y) and P(Y).

