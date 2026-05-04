import os
import json

def generate_image_pair_json(source_root, target_root):
    """
    从源文件夹的 trainA/trainB 子文件夹生成配对 JSON 文件（无中括号、单行单个对象）
    :param source_root: 源文件夹路径（包含 trainA、trainB 子文件夹）
    :param target_root: 目标文件夹路径（用于保存 JSON 文件）
    """
    # 1. 定义子文件夹路径
    trainA_path = os.path.join(source_root, "trainA")
    trainB_path = os.path.join(source_root, "trainB")
    
    # 2. 验证路径是否存在
    for path in [source_root, trainA_path, trainB_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在：{path}")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"不是有效文件夹：{path}")
    
    # 3. 获取 trainA、trainB 下的图片文件（仅保留文件名称，过滤目录）
    def get_image_filenames(folder_path):
        """获取文件夹下的图片文件名（过滤隐藏文件和目录）"""
        filenames = []
        for filename in os.listdir(folder_path):
            # 过滤隐藏文件（Ubuntu 下以 . 开头的文件）和非文件类型
            if not filename.startswith(".") and os.path.isfile(os.path.join(folder_path, filename)):
                # 仅保留常见图片格式（可根据需要扩展）
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    filenames.append(filename)
        return sorted(filenames)  # 排序保证匹配一致性
    
    trainA_files = get_image_filenames(trainA_path)
    trainB_files = get_image_filenames(trainB_path)
    
    # 4. 按文件名匹配，构建 JSON 数据列表
    json_data = []
    # 转换为集合快速查找匹配项
    trainB_file_set = set(trainB_files)
    for filename in trainA_files:
        if filename in trainB_file_set:
            # 严格按照你要求的 JSON 格式构建
            item = {
                "trainA": filename,
                "trainB": filename,
                "prompt": "an image"
            }
            json_data.append(item)
    
    if not json_data:
        print("警告：未找到匹配的图片文件对，生成的 JSON 将为空")
    
    # 5. 确保目标文件夹存在（不存在则创建）
    os.makedirs(target_root, exist_ok=True)
    
    # 6. 保存 JSON 文件（无中括号、单行单个对象）
    json_file_path = os.path.join(target_root, "image_pairs.json")
    with open(json_file_path, "w", encoding="utf-8") as f:
        # 遍历每个对象，单独序列化为单行字符串，逐行写入（无外层中括号）
        for item in json_data:
            # json.dumps 序列化单个对象，保证单行紧凑格式，无额外换行
            item_str = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            f.write(item_str + "\n")  # 每个对象一行，末尾换行分隔
    
    print(f"JSON 文件生成成功！")
    print(f"源文件夹：{source_root}")
    print(f"目标文件：{json_file_path}")
    print(f"匹配到图片对数量：{len(json_data)}")

if __name__ == "__main__":
    # ====================== 在这里修改为你的实际路径 ======================
    # 源文件夹路径（包含 trainA、trainB 子文件夹，支持绝对路径/相对路径）
    SOURCE_FOLDER = "/NAS_data/yjy/GF3_High_Res"  # Ubuntu 绝对路径示例
    # SOURCE_FOLDER = "./data/source"  # 相对路径示例（脚本所在目录下的子文件夹）
    
    # 目标文件夹路径（用于保存生成的 image_pairs.json）
    TARGET_FOLDER = "/data/hjf/TextDiff/ControlNet-main"  # Ubuntu 绝对路径示例
    # TARGET_FOLDER = "./data/target"  # 相对路径示例
    # =====================================================================
    
    # 执行生成逻辑并捕获异常
    try:
        generate_image_pair_json(SOURCE_FOLDER, TARGET_FOLDER)
    except Exception as e:
        print(f"错误：{type(e).__name__} - {str(e)}")