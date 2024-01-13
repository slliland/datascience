import pandas as pd
import matplotlib.pyplot as plt
import tkinter
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置加载的字体名
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_excel('/Users/songyujian/PycharmProjects/dataScience/middlereview.xlsx')


import jieba.analyse
jieba.analyse.set_stop_words('/Users/songyujian/PycharmProjects/dataScience/stopwords.txt')
#合并一起
text = ''
for i in range(len(df['cutword'])):
    text += str(df['cutword'][i])+'\n'
j_r=jieba.analyse.extract_tags(text,topK=20,withWeight=True)
df1 = pd.DataFrame()
df1['word']= [word[0] for word in j_r]  ;df1['frequency']=[word[1] for word in j_r]
print(df1)