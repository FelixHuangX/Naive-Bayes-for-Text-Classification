import jieba
import pandas as pd


# 函数要能对进来的.txt文件进行分词和去停用词，以及类别字符的编码，然后输出样本类标签，以及分词后的样本
def fenci(rawpath, stoppath):
    # 这里添加UTF-8的编码模式
    rawdata = open(rawpath, 'r', encoding="UTF-8")
    str_list = []
    line = rawdata.readline()
    # 将每一条新闻存为新闻类别以及新闻内容两个元素
    while line:
        line = line.strip('\n')
        line = line.split('\t')
        str_list.append(line)
        line = rawdata.readline()
    rawdata.close()

    Data = pd.DataFrame(str_list)
    data = Data[1]
    # 将新闻类别进行编码，此处有10类新闻，则编码为0-9
    data_target = list(pd.Series(Data[0].factorize()).iloc[0])

    stopwords = [line.strip('\n') for line in open(stoppath, 'r', encoding='utf-8').readlines()]

    n = len(data)
    result = []
    for i in range(n):
            # 对每一行的内容进行结巴分词
        Stence = jieba.lcut(data[i])
        result.append(movestopwords(Stence, stopwords))
        print(i)

    return result,data_target


# 对句子去除停用词
def movestopwords(sentence, stopwords):
    outstr = []
    for word in sentence:
        if word not in stopwords:
            if word != '\t' and '\n':
                outstr.append(word)
                # outstr += " "
    return outstr



