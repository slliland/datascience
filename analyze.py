import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置加载的字体名
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import jieba
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.globals import ThemeType

# 加载停用词表
stop_list  = pd.read_csv("/Users/songyujian/PycharmProjects/dataScience/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
# Jieba分词函数
def txt_cut(sentence):
    lis=[w for w in jieba.lcut(sentence) if w not in stop_list.values]
    return (" ").join(lis)
path = '/Users/songyujian/Documents'
df = pd.read_csv(path+'/reviews.csv')
df['cutword']=df['描述'].astype('str').apply(txt_cut)
df=df[['机型','日期','类别','描述','cutword']]
df=df.drop_duplicates(subset=['描述'])
df=df.fillna(method='backfill')
df.to_excel('/Users/songyujian/PycharmProjects/dataScience/newreview.xlsx',index=False)
df=df.reset_index(drop=True)
print(df.head())
print(df['机型'].value_counts())
print(df['类别'].value_counts())

# 将评论数据转换为文档储存
df = pd.read_excel('/Users/songyujian/PycharmProjects/dataScience/middlereview.xlsx')
model = df['机型'].value_counts()
model.to_csv("/Users/songyujian/PycharmProjects/dataScience/mmodel.csv",mode='a')
atype = df['类别'].value_counts()
atype.to_csv("/Users/songyujian/PycharmProjects/dataScience/matype.csv",mode='a')

# 读取分词之后的评论数据
mod = pd.read_csv('/Users/songyujian/PycharmProjects/dataScience/lmodel.csv')
mod_x = mod['机型']
mod_y = mod['count']
aty = pd.read_csv('/Users/songyujian/PycharmProjects/dataScience/latype.csv')
aty_x = aty['类别']
aty_y = aty['count']

x_data = mod_x
y_data = mod_y
data_pair = [list(z) for z in zip(x_data, y_data)]
data_pair.sort(key=lambda x: x[1])

# 绘制评论机型占比图
c = (
    # 宽  高  背景颜色
    Pie(init_opts=opts.InitOpts(width="1200px", height="800px",theme="dark"))
    .add(
        series_name="评论机型占比",    # 系列名称
        data_pair=data_pair,      # 系列数据项，格式为 [(key1, value1), (key2, value2)]
        rosetype="radius",        # radius：扇区圆心角展现数据的百分比，半径展现数据的大小
        radius="55%",             # 饼图的半径
        center=["50%", "50%"],    # 饼图的中心（圆心）坐标，数组的第一项是横坐标，第二项是纵坐标
        label_opts=opts.LabelOpts(is_show=False, position="center"),   #  标签配置项
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="机型分布",
            pos_left="center",
            pos_top="20",
            title_textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        legend_opts=opts.LegendOpts(type_='scroll', is_show=True, legend_icon='circle', selected_mode='multiple',
                                    orient="vertical", pos_top="20%", pos_right="10%"),
    )
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"  # 'item': 数据项图形触发，主要在散点图，饼图等无类目轴的图表中使用
         ),
        label_opts=opts.LabelOpts(color="rgba(255, 255, 255, 0.3)"),
    )
    .render("customized_pie.html")
)

# 绘制评论类型占比图
d = (
    Pie(init_opts=opts.InitOpts(width="1700px",
                                height="800px",
                                theme="dark"))
    .add(
        "",
        [list(z) for z in zip(aty_x, aty_y)],
        # 饼图的半径，数组的第一项是内半径，第二项是外半径
        # 默认设置成百分比，相对于容器高宽中较小的一项的一半
        radius=["40%", "60%"],
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="评论类型"),
        legend_opts=opts.LegendOpts(type_='scroll',is_show=True,legend_icon='circle',selected_mode='multiple',orient="vertical", pos_top="0%", pos_right="10%"),
    )
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    .render("pie_radius.html")
)