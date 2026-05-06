# 导入必要的库
# import json  # 用于解析JSON格式的数据文件
# import cv2   # OpenCV库，用于图像读取和处理
# import numpy as np  # NumPy库，用于数值计算和数组操作
import json
import os
import cv2
import numpy as np
# 导入PyTorch的数据集基类
from torch.utils.data import Dataset
# from tutorial_dataset import MyDataset  # 注释掉的导入，可能是避免循环导入

# 定义自定义数据集类，继承自PyTorch的Dataset基类
class MyDataset(Dataset):
    def __init__(
        self,
        json_path='/data/hjf/TextDiff/ControlNet-main/image_pairs.json',
        source_root='/NAS_data/yjy/GF3_High_Res/trainA',
        target_root='/NAS_data/yjy/GF3_High_Res/trainB',
        color_hint_root=None,
        mask_root=None,
        image_size=256,
        default_prompt='an image',
    ):
        self.json_path = json_path
        self.source_root = source_root
        self.target_root = target_root
        self.color_hint_root = color_hint_root
        self.mask_root = mask_root
        self.image_size = image_size
        self.default_prompt = default_prompt
        # 初始化数据列表，用于存储从JSON文件读取的数据
        self.data = []
        # 注释掉的原数据路径：fill50k数据集
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        # 打开包含图像对信息的JSON文件，使用文本模式读取
        with open(self.json_path, 'rt', encoding='utf-8') as f:
            # 逐行读取JSON文件
            for line in f:
                # 将每行JSON字符串解析为Python字典并添加到数据列表中
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        # 返回数据集的大小，PyTorch DataLoader需要这个方法
        return len(self.data)

    # def __getitem__(self, idx):
    #     # 根据索引获取单个数据样本，PyTorch DataLoader会调用这个方法
    #     # 从数据列表中获取指定索引的数据项
    #     item = self.data[idx]
    def _resolve_path(self, root, filename):
        if filename is None:
            return None
        if os.path.isabs(filename):
            return filename
        if root is None:
            return filename
        return os.path.join(root, filename)

    def _read_rgb(self, path, label):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Failed to read {label} image: {path}')
        image = cv2.resize(image, (self.image_size, self.image_size))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Failed to read mask image: {path}')
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.float32) / 255.0

    def _load_color_hint(self, item, source_filename):
        color_hint_filename = item.get('color_hint', source_filename)
        mask_filename = item.get('mask', color_hint_filename)
        color_hint_path = self._resolve_path(self.color_hint_root, color_hint_filename)
        mask_path = self._resolve_path(self.mask_root, mask_filename)

        color_hint = self._read_rgb(color_hint_path, 'color_hint').astype(np.float32) / 255.0
        mask = self._read_mask(mask_path)[..., None]
        return np.concatenate([color_hint, mask], axis=2).astype(np.float32)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['trainA']
        target_filename = item['trainB']
        prompt = item.get('prompt', self.default_prompt)

        source = self._read_rgb(self._resolve_path(self.source_root, source_filename), 'source')
        target = self._read_rgb(self._resolve_path(self.target_root, target_filename), 'target')

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0



        sample = dict(jpg=target, txt=prompt, hint=source)
        if self.color_hint_root is not None and self.mask_root is not None:
            sample['color_hint'] = self._load_color_hint(item, source_filename)
        return sample

if __name__ == '__main__':
    dataset = MyDataset()
    print(len(dataset))
    item = dataset[0]
    print(item['txt'])
    print(item['jpg'].shape)
    print(item['hint'].shape)
    if 'color_hint' in item:
        print(item['color_hint'].shape)
