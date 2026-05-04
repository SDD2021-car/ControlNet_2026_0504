import os
import shutil
import re
from tqdm import tqdm  # 进度条库，Ubuntu下需先安装

# ====================== 可直接编辑的路径（修改这里！） ======================
# Ubuntu路径格式示例：/home/你的用户名/input_files 或 /mnt/data/checkpoints
INPUT_DIR = "/data/hjf/TextDiff/ControlNet-main/checkpoints/season"  # 例：/home/user/checkpoints/sar2Opt
OUTPUT_DIR = "/NAS_data/hjf/ControlNet-main/checkpoints/season"  # 例：/home/user/checkpoints/old_files
# ===========================================================================

def main():
    # 验证输入文件夹是否存在（Ubuntu路径区分大小写！）
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ 错误：输入文件夹「{INPUT_DIR}」不存在，请检查路径（注意大小写）！")
        return

    # 创建输出文件夹（已存在则不报错，自动处理权限）
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"✅ 输出文件夹已就绪：{OUTPUT_DIR}\n")
    except PermissionError:
        print(f"❌ 权限不足！无法创建输出文件夹「{OUTPUT_DIR}」，请加sudo运行脚本")
        return

    # 正则表达式：匹配文件名中的数字（根据你的实际文件名格式调整！）
    # 示例匹配：step_step=1234.ckpt → 提取1234；若文件名是file_567.txt，需改为 r'file_(\d+)\.txt$'
    num_regex = re.compile(r'step_step=(\d+)\.ckpt$')
    file_info = []

    # 1. 扫描输入文件夹（仅处理文件，跳过子文件夹）
    print("🔍 正在扫描输入文件夹中的文件...")
    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        if os.path.isfile(file_path):
            match = num_regex.search(filename)
            if match:
                file_num = int(match.group(1))
                file_info.append((file_num, file_path, filename))

    # 无符合条件文件的处理
    if not file_info:
        print("⚠️ 未找到符合格式的文件（当前匹配规则：step_step=数字.ckpt）")
        return

    # 2. 筛选保留数字最大的文件（最新）
    file_info.sort(key=lambda x: x[0], reverse=True)
    keep_file = file_info[0]
    print(f"\n📌 保留最新文件：{keep_file[2]}（数字：{keep_file[0]}）")

    # 3. 提取需要移动的文件
    files_to_move = file_info[1:]
    total_files = len(files_to_move)
    
    if total_files == 0:
        print("\n🎉 没有需要移动的文件（仅保留了最新文件）")
        return

    # 4. 带进度条移动文件（捕获Ubuntu下的常见错误）
    print(f"\n🚀 开始移动 {total_files} 个文件到输出文件夹...")
    for _, (file_num, src_path, filename) in enumerate(tqdm(
        files_to_move, 
        desc="移动进度", 
        unit="文件", 
        ncols=80
    ), 1):
        try:
            dest_path = os.path.join(OUTPUT_DIR, filename)
            # Ubuntu下移动文件，若目标已存在则先删除（可选，根据需求）
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(src_path, dest_path)
        except PermissionError:
            print(f"\n❌ 权限不足！无法移动 {filename}，请加sudo运行")
            return
        except Exception as e:
            print(f"\n❌ 移动文件失败：{filename} → 错误：{str(e)}")

    print(f"\n✅ 全部完成！共移动 {total_files} 个文件到：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()