from gensim import corpora,models,similarities,utils
import os
f = open(r'alltext\绑架罪\000.txt',encoding='utf-8')
lines = f.readlines()
text = []
anyou = ''
for line in lines:#将文件中的词装入列表
     if len(line) != 1:
            # 抽掉引用法条
            if line.split('##')[0] != '引用法条' and line.split('##')[0] != '特征' and line.split('##')[0] != '案由':
                for word in line.split('##')[1].split('$'):
                    text.append(word.replace('\n', '').replace('\t', ''))
            if line.split('##')[0] == '特征':
                for word in line.split('##')[1].split('\t'):
                    text.append(word.replace('\n', '').replace('\t', ''))
            if line.split('##')[0] == '案由':
                for word in line.split('##')[1].split('\t'):
                    text.append(word.replace('\n', '').replace('\t', ''))
#到该案由路径下载入获取已经训练好的模型
fr = r'result'
output = os.path.join (fr,'绑架罪')
# 载入字典
dictionary = corpora.Dictionary.load(os.path.join(output,"all.dic"))
# 载入TFIDF模型和索引
tfidfModel = models.TfidfModel.load(os.path.join(output,"allTFIDF.mdl"))
indexTfidf = similarities.MatrixSimilarity.load(os.path.join(output,"allTFIDF.idx"))
# 载入LDA模型和索引
ldaModel = models.LdaModel.load(os.path.join(output,  "allLDA50Topic.mdl"))
indexLDA = similarities.MatrixSimilarity.load(os.path.join (output,"allLDA50Topic.idx"))
#query就是测试数据，先切词
query_bow = dictionary.doc2bow(text)
#使用TFIDF模型向量化
tfidfvect = tfidfModel[query_bow]
#然后LDA向量化，因为我们训练时的LDA是在TFIDF基础上做的，所以用itidfvect再向量化一次
ldavec = ldaModel[tfidfvect]
#TFIDF相似性，相似性列表记录与每份模型中的文书的相似度
simstfidf = indexTfidf[tfidfvect]
#LDA相似性，相似性列表记录与每份模型中的文书的相似度
simlda = indexLDA[ldavec]
# 获取TDIDF模型中与该文档前十相近的文书号
# 记录序号
num = []
# 记录相似度
gets = []
for i in range(0,10):
    # 相似度
    m = 0
    # 文书序号
    mark = 0
    for likely in simstfidf:
        if m < likely and likely not in gets:
                m = likely
                marknow = mark
        mark = mark+1
    gets.append(m)
    num.append(marknow)
print(num)

# 获取LDA模型中与该文档前十相近的文书号
# 记录序号
numlda = []
# 记录相似度
getslda = []
for i in range(0,10):
    # 相似度
    m = 0
    # 文书序号
    mark = 0
    for likely in simlda:
        if m < likely and likely not in getslda:
            m = likely
            marknow = mark
        mark = mark+1
    getslda.append(m)
    numlda.append(marknow)
print(numlda)
