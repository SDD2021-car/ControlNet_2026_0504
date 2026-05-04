# 导入必要的库
import json  # 用于解析JSON格式的数据文件
import cv2   # OpenCV库，用于图像读取和处理
import numpy as np  # NumPy库，用于数值计算和数组操作

# 导入PyTorch的数据集基类
from torch.utils.data import Dataset
# from tutorial_dataset import MyDataset  # 注释掉的导入，可能是避免循环导入

# 定义自定义数据集类，继承自PyTorch的Dataset基类
class MyDataset(Dataset):
    def __init__(self):
        # 初始化数据列表，用于存储从JSON文件读取的数据
        self.data = []
        # 注释掉的原数据路径：fill50k数据集
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        # 打开包含图像对信息的JSON文件，使用文本模式读取
        with open('/data/hjf/TextDiff/ControlNet-main/image_pairs.json', 'rt') as f:
            # 逐行读取JSON文件
            for line in f:
                # 将每行JSON字符串解析为Python字典并添加到数据列表中
                self.data.append(json.loads(line))

    def __len__(self):
        # 返回数据集的大小，PyTorch DataLoader需要这个方法
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引获取单个数据样本，PyTorch DataLoader会调用这个方法
        # 从数据列表中获取指定索引的数据项
        item = self.data[idx]

        # 从数据项中提取文件名和提示文本
        source_filename = item['trainA']  # 控制条件图像的文件名
        target_filename = item['trainB']  # 目标图像的文件名
        prompt = item['prompt']           # 文本提示描述

        # 注释掉的原数据路径：fill50k数据集的图像读取
        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        # 读取控制条件图像（SAR雷达图像），从trainA目录
        source = cv2.imread('/NAS_data/yjy/GF3_High_Res/trainA/'+ source_filename)
        # 读取目标图像（光学图像），从trainB目录
        target = cv2.imread('/NAS_data/yjy/GF3_High_Res/trainB/' +target_filename)
        
        # 将图像调整到ControlNet要求的512x512尺寸
        source = cv2.resize(source, (256, 256))
        target = cv2.resize(target, (256, 256))
        
        # 重要：OpenCV默认以BGR格式读取图像，需要转换为RGB格式
        # 因为深度学习模型通常使用RGB格式
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # 将控制条件图像归一化到[0, 1]范围
        # 先转换为float32类型，然后除以255
        source = source.astype(np.float32) / 255.0

        # 将目标图像归一化到[-1, 1]范围
        # 这是Stable Diffusion模型的标准输入范围
        # 先除以127.5，然后减去1.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        # 返回训练所需的数据字典
        # jpg: 目标图像，用于训练生成
        # txt: 文本提示，用于条件生成
        # hint: 控制条件图像，用于控制生成过程
        return dict(jpg=target, txt=prompt, hint=source)




# 测试代码：创建数据集实例并查看数据
dataset = MyDataset()  # 实例化数据集
print(len(dataset))    # 打印数据集大小

# 获取第1234个样本进行测试
item = dataset[1234]
# 提取数据字典中的各个组件
jpg = item['jpg']      # 目标图像
txt = item['txt']      # 文本提示
hint = item['hint']    # 控制条件图像

# 打印测试结果
print(txt)             # 打印文本提示
print(jpg.shape)       # 打印目标图像的形状
print(hint.shape)      # 打印控制条件图像的形状
