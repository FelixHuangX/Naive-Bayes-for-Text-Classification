from numpy import *
from participle import *
from Naivebayes_train import loadDataSet,words2vec

#朴素贝叶斯分类器
def classifyNaiveB(doc2classify,pVec,ptarget):

    p = []
    for i in range(len(ptarget)):
        # aa = doc2classify*pVec[i]
        # bb = pVec[i]
        # cc = sum(doc2classify*pVec[i])
        prate = sum(doc2classify*pVec[i]) + log(ptarget[i]) #取对数后变为“+”号
        p.append(prate)
    return p.index(max(p))


if __name__ == "__main__":
    #先导入数据所在的路径
    stoppath = r'./cnews/cnews.vocab.txt'
    testpath = r'./cnews/cnews.test.txt'

    #导入训练好的模型
    pos = list(load('models/pos.npy'))
    UniquesetList = [line.strip('\n') for line in open('models/UniquesetList.txt', 'r').readlines()]
    pV = load('models/pV.npy')


    #测试，对测试集进行处理
    testEntry, testclasses = loadDataSet(testpath, stoppath)
    testmat = []
    for singlefeatures in testEntry:
        testmat.append(words2vec(UniquesetList, singlefeatures))
        print(testEntry.index(singlefeatures))

    result = []
    for i in range(len(testmat)):
        a = classifyNaiveB(testmat[i],pV,pos)
        result.append(a)

    #比较预测标签与真实标签的的匹配率，即为分类的正确率
    ac = 0
    for i in range(len(result)):
        if testclasses[i] == result[i]:
            ac += 1

    accuracy = ac/len(result)
    print('Test Accuracy:',accuracy)