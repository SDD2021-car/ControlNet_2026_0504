import json

input_jsonl = "/data/hjf/TextDiff/datasets/scene/metadata.jsonl"   # 原始 jsonl 文件
output_jsonl = "/data/hjf/TextDiff/ControlNet-main/SCENE.json"   # 输出 json 文件

with open(input_jsonl, "r", encoding="utf-8") as fin, \
     open(output_jsonl, "w", encoding="utf-8") as fout:

    for line in fin:
        if not line.strip():
            continue
        item = json.loads(line)
        item["text"] = "an image"
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
