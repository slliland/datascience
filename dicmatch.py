import inline as inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba.posseg as psg
from wordcloud import WordCloud

word = pd.read_csv("/Users/songyujian/PycharmProjects/dataScience/word.csv")

# 读入正面、负面情感评价词
pos_comment = pd.read_csv('/Users/songyujian/PycharmProjects/dataScience/正面情感词语（中文）.txt', header=None,sep="/n",
                          encoding = 'utf-8', engine='python')
neg_comment = pd.read_csv('/Users/songyujian/PycharmProjects/dataScience/正面评价词语（中文）.txt', header=None,sep="/n",
                          encoding = 'utf-8', engine='python')
pos_emotion = pd.read_csv('/Users/songyujian/PycharmProjects/dataScience/负面情感词语（中文）.txt', header=None,sep="/n",
                          encoding = 'utf-8', engine='python')
neg_emotion = pd.read_csv('/Users/songyujian/PycharmProjects/dataScience/负面评价词语（中文）.txt', header=None,sep="/n",
                          encoding = 'utf-8', engine='python')

# 合并情感词与评价词
positive = set(pos_comment.iloc[:,0])|set(pos_emotion.iloc[:,0])
negative = set(neg_comment.iloc[:,0])|set(neg_emotion.iloc[:,0])

# 正负面情感词表中相同的词语
intersection = positive&negative

positive = list(positive - intersection)
negative = list(negative - intersection)

positive = pd.DataFrame({"词":positive,
                         "权重":[1]*len(positive)})
negative = pd.DataFrame({"词":negative,
                         "权重":[-1]*len(negative)})

posneg = pd.concat([positive, negative], ignore_index=True)
print(posneg)

# 将分词结果与正负面情感词表合并，定位情感词
data_posneg = posneg.merge(word, left_on = '词', right_on = '词',
                           how = 'right')
data_posneg = data_posneg.sort_values(by = ['序号','位置'])

print(data_posneg.head())

# 载入否定词表
notdict = pd.read_csv('/Users/songyujian/PycharmProjects/dataScience/否定词.txt')

# 构造新列，作为经过否定词修正后的情感值
data_posneg['修正权重'] = data_posneg['权重']
data_posneg['id'] = np.arange(0, len(data_posneg))

# 只保留有情感值的词语
only_inclination = data_posneg.dropna().reset_index(drop=True)

index = only_inclination['id']

for i in np.arange(0, len(only_inclination)):
    # 提取第i个情感词所在的评论
    review = data_posneg[data_posneg['序号'] == only_inclination['序号'][i]]
    review.index = np.arange(0, len(review))
    # 第i个情感值在该文档的位置
    affective = only_inclination['位置'][i]
    if affective == 1:
        ne = sum([i in notdict['term'] for i in review['词'][affective - 1]]) % 2
        if ne == 1:
            data_posneg['修正权重'][index[i]] = -data_posneg['权重'][index[i]]
    elif affective > 1:
        ne = sum([i in notdict['term'] for i in review['词'][[affective - 1,
                                                                affective - 2]]]) % 2
        if ne == 1:
            data_posneg['修正权重'][index[i]] = -data_posneg['权重'][index[i]]

# 更新只保留情感值的数据
only_inclination = only_inclination.dropna()

# 计算每条评论的情感值
emotional_value = only_inclination.groupby(['序号'],
                                           as_index=False)['修正权重'].sum()

# 去除情感值为0的评论
emotional_value = emotional_value[emotional_value['修正权重'] != 0]
# 给情感值大于0的赋予评论类型（content_type）为pos,小于0的为neg
emotional_value['a_type'] = ''
emotional_value['a_type'][emotional_value['修正权重'] > 0] = 'pos'
emotional_value['a_type'][emotional_value['修正权重'] < 0] = 'neg'

print(emotional_value.head())

# 查看情感分析结果
result = emotional_value.merge(word,
                               left_on = '序号',
                               right_on = '序号',
                               how = 'left')
print(result.head())
result = result[['序号','机型', 'a_type']].drop_duplicates()
print(result.head())
# 交叉表:统计分组频率的特殊透视表
confusion_matrix = pd.crosstab(result['机型'], result['a_type'],
                               margins=True)
print(confusion_matrix.head())
# 提取正负面评论信息
ind_pos = list(emotional_value[emotional_value['a_type'] == 'pos']['序号'])
ind_neg = list(emotional_value[emotional_value['a_type'] == 'neg']['序号'])
posdata = word[[i in ind_pos for i in word['序号']]]
negdata = word[[i in ind_neg for i in word['序号']]]
# 正面情感词词云
freq_pos = posdata.groupby('词')['词'].count()
freq_pos = freq_pos.sort_values(ascending = False)
font_path =  "/System/Library/Fonts/PingFang.ttc"
backgroud_Image=plt.imread('/Users/songyujian/PycharmProjects/dataScience/bg.png')
wordcloud = WordCloud(font_path=font_path,
                      max_words=100,
                      background_color='white',
                      mask=backgroud_Image)
pos_wordcloud = wordcloud.fit_words(freq_pos)
plt.imshow(pos_wordcloud)
plt.axis('off')
plt.show()


# 负面情感词词云
freq_neg = negdata.groupby(by = ['词'])['词'].count()
freq_neg = freq_neg.sort_values(ascending = False)
neg_wordcloud = wordcloud.fit_words(freq_neg)
plt.imshow(neg_wordcloud)
plt.axis('off')
plt.show()
# 将结果写出,每条评论作为一行
posdata.to_csv("/Users/songyujian/PycharmProjects/dataScience/posdata.csv", index = False, encoding = 'utf-8')
negdata.to_csv("/Users/songyujian/PycharmProjects/dataScience/negdata.csv", index = False, encoding = 'utf-8')