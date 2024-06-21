# import matplotlib.pyplot as plt
#
# # 读取f1_scores.txt文件
# file_path = '../out/single_output7.txt'
#
# # 初始化一个空列表来存储F1分数
# f1_scores = []
#
# # 读取文件并将每一行的F1分数添加到列表中
# with open(file_path, 'r') as file:
#     for line in file:
#         score = float(line.strip())  # 将每一行的F1分数转换为浮点数并去除换行符
#         f1_scores.append(score)
#
# # 只取前300轮的F1分数
# f1_scores = f1_scores[:300]
#
# # 创建一个包含轮次的列表
# rounds = list(range(1, len(f1_scores) + 1))
#
# # 绘制折线图
# plt.figure(figsize=(10, 5))
# plt.plot(rounds, f1_scores, marker='', linestyle='-', color='b')
#
# # 添加标题和标签
# plt.title('F1 Score over First 300 Rounds')
# plt.xlabel('Round')
# plt.ylabel('F1 Score')
#
# # 显示网格
# plt.grid(False)
#
# # 显示图形
# plt.show()
import matplotlib.pyplot as plt

# 读取f1_scores.txt文件
file_path = '../out/single_output8.txt'
file_path1 = '../out/single_output7.txt'
file_path2 = '../out/single_outputconv.txt'
file_path3 = '../out/single_outputgoole.txt'
# 初始化一个空列表来存储F1分数
f1_scores = []
f1_scores1 = []
f1_scores2 = []
f1_scores3 = []

# 读取文件并将每一行的F1分数添加到列表中
with open(file_path, 'r') as file:
    for line in file:
        score = float(line.strip())  # 将每一行的F1分数转换为浮点数并去除换行符
        f1_scores.append(score)
with open(file_path1, 'r') as file:
    for line1 in file:
        score1 = float(line1.strip())  # 将每一行的F1分数转换为浮点数并去除换行符
        f1_scores1.append(score1)
with open(file_path2, 'r') as file:
    for line2 in file:
        score2 = float(line2.strip())  # 将每一行的F1分数转换为浮点数并去除换行符
        f1_scores2.append(score2)
with open(file_path3, 'r') as file:
    for line3 in file:
        score3 = float(line3.strip())  # 将每一行的F1分数转换为浮点数并去除换行符
        f1_scores3.append(score3)


# 只取前300轮的F1分数
f1_scores = f1_scores[:800]
f1_scores1 = f1_scores1[:800]
f1_scores2 = f1_scores2[:800]
f1_scores3 = f1_scores3[:800]
# 创建一个包含轮次的列表
rounds = list(range(1, len(f1_scores) + 1))
rounds1 = list(range(1, len(f1_scores1) + 1))
rounds2 = list(range(1, len(f1_scores2) + 1))
rounds3 = list(range(1, len(f1_scores3) + 1))
# 过滤F1分数在0.8到1之间的数据
filtered_f1_scores = []
filtered_rounds = []
filtered_f1_scores1 = []
filtered_rounds1 = []
filtered_f1_scores2 = []
filtered_rounds2 = []
filtered_f1_scores3 = []
filtered_rounds3 = []
for i, score in enumerate(f1_scores):
    # if 0.5 <= score <= 1:
        filtered_f1_scores.append(score)
        filtered_rounds.append(rounds[i])
for i, score1 in enumerate(f1_scores1):
    # if 0.5 <= score1 <= 1:
        filtered_f1_scores1.append(score1)
        filtered_rounds1.append(rounds1[i])
for i, score2 in enumerate(f1_scores2):
    # if 0.5 <= score2 <= 1:
        filtered_f1_scores2.append(score2)
        filtered_rounds2.append(rounds2[i])
for i, score3 in enumerate(f1_scores3):
    # if 0.5 <= score3 <= 1:
        filtered_f1_scores3.append(score3)
        filtered_rounds3.append(rounds3[i])


# 绘制折线图
plt.figure(figsize=(10, 5))
plt.plot(filtered_rounds, filtered_f1_scores, marker='', linestyle='-', color='b', label='Resnet-CMH')
plt.plot(filtered_rounds1, filtered_f1_scores1, marker='', linestyle='-', color='r', label='ResNet-18')
plt.plot(filtered_rounds2, filtered_f1_scores2, marker='', linestyle='-', color='y', label='ConvNeXt')
plt.plot(filtered_rounds3, filtered_f1_scores3, marker='', linestyle='-', color='g', label='GoogLeNet')

# 添加标题和标签
plt.title('F1 Score (0 - 1)')
plt.xlabel('epoch')
plt.ylabel('F1 Score')

# 添加图例
plt.legend()

# 显示网格
plt.grid(False)

# 显示图形
plt.show()