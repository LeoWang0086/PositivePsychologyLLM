import plotly.graph_objects as go

# 定义策略节点
nodes = ["Approval & Reassurance", "Interpretation", "Direct Guidance", 
         "Information", "Self-disclosure", "Restatement"]

# 定义"源"、"目标"和"值" (转换频率)
source = [0, 1, 3, 0, 2]  # 对应节点索引
target = [1, 2, 0, 4, 1]  # 对应的目标节点索引
values = [15, 10, 8, 12, 5]  # 转换的次数

# 创建桑基图
fig = go.Figure(go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = nodes
    ),
    link = dict(
        source = source,
        target = target,
        value = values
    )
))

# 展示图表
fig.show()
