import inline as inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba.posseg as psg
from wordcloud import WordCloud


import warnings
warnings.filterwarnings("ignore")

path = '/Users/songyujian/Documents'
reviews = pd.read_csv(path+'/reviews.csv')
# print(reviews.shape)
reviews.head()
# 删除描述、机型两栏内容完全相同的评论
reviews = reviews[['描述','机型']].drop_duplicates()
content = reviews['描述']
print(reviews.shape)
# 去除英文、数字、评论大量出现的中性词等词语
info = re.compile('[0-9a-zA-Z]|手机|使用|时间|用户|设置|现在|表示|提供|希望|反馈|是否|一下|时|问题|出现')
content = content.apply(lambda x: info.sub('',str(x)))
# 分词
worker = lambda s: [(x.word, x.flag) for x in psg.cut(s)] # 自定义简单分词函数
seg_word = content.apply(worker)
# print(seg_word.head())
# 将词语转为数据框形式，一列是词，一列是词语所在的句子ID，最后一列是词语在该句子的位置
n_word = seg_word.apply(lambda x: len(x))  # 每一评论中词的个数

n_content = [[x+1]*y for x,y in zip(list(seg_word.index), list(n_word))]

# 将嵌套的列表展开，作为词所在评论的id
index_content = sum(n_content, [])

seg_word = sum(seg_word, [])
# 词
word = [x[0] for x in seg_word]
# 词性
nature = [x[1] for x in seg_word]

content_type = [[x]*y for x,y in zip(list(reviews['机型']), list(n_word))]
# 评论类型
content_type = sum(content_type, [])

result = pd.DataFrame({"序号":index_content,
                       "词":word,
                       "词性":nature,
                       "机型":content_type})
# print(result.head())
# 删除标点符号
result = result[result['词性'] != 'x']  # x表示标点符号

# 删除停用词
stop_path = open("/Users/songyujian/PycharmProjects/dataScience/stopwords.txt", 'r',encoding='UTF-8')
stop = stop_path.readlines()
stop = [x.replace('\n', '') for x in stop]
word = list(set(word) - set(stop))
result = result[result['词'].isin(word)]
print(result.head())
# 构造各词在对应评论的位置列
n_word = list(result.groupby(by = ['序号'])['序号'].count())
index_word = [list(np.arange(0, y)) for y in n_word]
# 词语在该评论的位置
index_word = sum(index_word, [])
# 合并评论id
result['位置'] = index_word
print(result.head())
# 提取含有名词类的评论,即词性含有“n”的评论
ind = result[['n' in x for x in result['词性']]]['序号'].unique()
result = result[[x in ind for x in result['序号']]]
print(result.head())
frequencies = result.groupby('词')['词'].count()
frequencies = frequencies.sort_values(ascending = False)
backgroud_Image=plt.imread("/Users/songyujian/PycharmProjects/dataScience/bg.png")

# 自己上传中文字体到kesci
font_path =  "/System/Library/Fonts/PingFang.ttc"
wordcloud = WordCloud(font_path=font_path, # 设置字体，不设置就会出现乱码
                      max_words=100,
                      background_color='white',
                      mask=backgroud_Image)# 词云形状

my_wordcloud = wordcloud.fit_words(frequencies)
plt.imshow(my_wordcloud)
plt.axis('off')
plt.show()
# 将结果保存
result.to_csv("/Users/songyujian/PycharmProjects/dataScience/word.csv", index = False, encoding = 'utf-8')