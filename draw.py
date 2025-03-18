import matplotlib.pyplot as plt
import numpy as np
def sum_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [float(line.strip()) for line in lines]
    return data, sum(data)
# 读取txt文件中的数据
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 假设每一行的数字都是单个浮动数字
        data = [float(line.strip()) for line in lines]
    return data

# 绘制多文件数据，确保横坐标一致
def plot_multiple_files(files):
    # 先读取所有文件的数据并找出最大的行数
    all_data = []
    max_length = 0
    for file_path in files:
        data = read_data_from_file(file_path)
        all_data.append(data)
        max_length = max(max_length, len(data))

    # 绘图
    labels = ["self-refine", "MAD"]
    plt.figure(figsize=(10, 6))
    for i, data in enumerate(all_data):
        # 如果数据的长度小于最大长度，填充 NaN
        x = range(1, max_length + 1)
        y = data + [np.nan] * (max_length - len(data))  # 填充NaN到较短的文件
        
        plt.plot(x, y, label=labels[i])  # 为每个文件的数据加标签
    
    plt.xlabel('Number of generation')
    plt.ylabel('Number of tokens')
    # plt.title('Number of generation')
    plt.legend()  # 添加图例，区分不同数据
    plt.grid(True)  # 添加网格
    # 将图形保存为文件
    plt.savefig('/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/output_plot2.png')
    # 显示图形
    plt.show()

# 主函数
if __name__ == "__main__":
    # 这里列出所有需要显示的文件路径
    files = ["/project/pi_hongyu_umass_edu/zonghai/hospital_translation/self-refine/output_gpt_5.txt", "/project/pi_hongyu_umass_edu/zonghai/hospital_translation/Multi-Agents-Debate/output.txt"]  # 替换为实际的txt文件路径
    plot_multiple_files(files)
    data1, sum1 = sum_from_file(file1_path)
    data2, sum2 = sum_from_file(file2_path)

    print(f"File1 ({file1_path}) sum: {sum1}")
    print(f"File2 ({file2_path}) sum: {sum2}")
