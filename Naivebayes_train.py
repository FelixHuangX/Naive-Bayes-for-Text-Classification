
from numpy import *
from participle import *
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#对新闻数据进行分类
def loadDataSet(rawpath, stoppath):
    result, data_target = fenci(rawpath, stoppath)
    return result, data_target


#创建一个包含所有文档中出现的不重复词的列表
def createUniqueSet(dataset):
    #先创建一个空集
    UniqueSet = {}
    for i in range(len(dataset)):
    #for document in dataset:
        #创建集合的并集,同时对词频数较小的词去除以实现降维
        for word in dataset[i]:
            if word in UniqueSet:
                UniqueSet[word] += 1
            else:
                UniqueSet[word] = 1
        #UniqueSet = UniqueSet | set(dataset[i])
        print(i)

    #设置最小词频数，将健的值小于最小词频数的健去掉
    UniqueSet = {k: v for k, v in UniqueSet.items() if v > 50}


    items = list(UniqueSet.items())  # 将键值对转换成列表
    items.sort(key=lambda x: x[1], reverse=True)


    return list(UniqueSet.keys())

#word2vec,将文档词条转化为词向量
def words2vec(UniquesetList,inputdoc):
    returnVec = [0]*len(UniquesetList)
    for word in inputdoc:
        if word in UniquesetList:
            #文档的词袋模型，每个词可以出现多次
            returnVec[UniquesetList.index(word)] += 1
        # else:
        #     print('the word: %s is not in my vocabulary!')
    return returnVec

#朴素贝叶斯分类器训练函数，从词向量计算概率
def trainNaiveB(trainMat,trainClass):
    #估计P(Y)以及P(X|Y)
    numTraindocs = len(trainMat)
    numWords = len(trainMat[0])
    dataclass = set(trainClass)
    ptarget = []
    PVec = []
    for i in range(len(dataclass)):
        posSampleRate = trainClass.count(i)/float(numTraindocs)
        ptarget.append(posSampleRate)
        # 初始化两类数据中的各词频数，以及两类数据中词的总数
        # 不取0是为了避免下溢出
        pNum = ones(numWords)
        pSum = 2.0
        for j in range(numTraindocs):
            if trainClass[j] == i:
                pNum += trainMat[j]
                pSum += sum(trainMat[j])
        print(i)
        pVec = log(pNum / pSum)  # 取对是为了避免下溢出或者浮点数舍入导致的错误，下溢出是由太多很小的数相乘得到
        PVec.append(pVec)

    return PVec,ptarget





if __name__ == "__main__":
    label = ['Sport','Entertainment','Household','House Property','Education','Fashion','Current Politics','Game','Science and Technology','Finance']
    #先导入数据所在的路径
    rawpath = r'./cnews/cnews.val.txt'
    stoppath = r'./cnews/cnews.vocab.txt'
    trainpath = r'./cnews/cnews.train.txt'
    testpath = r'./cnews/cnews.test.txt'
    #对原数据进行分词，去除停用词，以及对中文标签进行编码并返回
    rawfeatures,rawclasses = loadDataSet(trainpath, stoppath)

    #对每个文本分词的结果转化为词向量
    UniquesetList = createUniqueSet(rawfeatures)
    trainmat = []
    for singlefeatures in rawfeatures:
        trainmat.append(words2vec(UniquesetList,singlefeatures))
        print(rawfeatures.index(singlefeatures))

    #训练模型
    pV,pos = trainNaiveB(array(trainmat),rawclasses)

    #将模型的训练结果存储下来便于测试文件调用
    numpy_array1 = array(pos)
    save('models/pos.npy',numpy_array1)

    file = open('models/UniquesetList.txt','w')
    for i in range(len(UniquesetList)):
        file.write(UniquesetList[i])
        file.write('\n')
    file.close()

    numpy_array2 = array(pV)
    save('models/pV.npy', numpy_array2)

    #词云
    # npclasses = array(rawclasses)
    # font = r'font/simhei.ttf'
    # for i in set(rawclasses):
    #     plt.figure(i)
    #     tindex = where(npclasses == i)
    #     temfeatures = rawfeatures[tindex[0][0]:tindex[0][len(tindex[0])-1]+1]
    #     tf = {}  # 统计词典
    #     for seg in temfeatures:
    #         for j in seg:
    #             if j in tf:   #如果该键在集合的对象中，则该键所属对象值加1
    #                 tf[j] += 1
    #             else:   #否则，生成新词的键值对，初始值为1
    #                 tf[j] = 1
    #     tf = {k: v for k, v in tf.items() if v > 30}
    #     wc=WordCloud(collocations=False, font_path=font, max_font_size=200, width=1600, margin=0).generate_from_frequencies(tf)
    #
    #     plt.imshow(wc)
    #     plt.axis('off')
    #     plt.title(label[i]+' News Wordcloud')
    # plt.show()
    #plt.close(all)