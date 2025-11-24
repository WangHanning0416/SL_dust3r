import os
import json

def load_counter():
    counter_file = "counter.txt"
    if os.path.exists(counter_file):
        with open(counter_file, 'r', encoding='utf-8') as f:
            try:
                return int(f.read().strip())
            except:
                return 1
    else:
        return 1

def save_counter(num):
    with open("counter.txt", 'w', encoding='utf-8') as f:
        f.write(str(num))

def extract_metric(scene):
    json_path = os.path.join("result", scene, "eval", "recon_metric_local.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        metric_str = data.get("metric", "")
        metrics = {}
        for item in metric_str.split(", "):
            if ":" in item:
                key, value = item.split(":", 1)
                metrics[key.strip()] = value.strip()
        acc = metrics.get("Acc")
        comp = metrics.get("Comp")
        return f"{acc}/{comp}" if acc and comp else "N/A"
    except:
        return "N/A"

def get_max_column_widths(all_rows):
    """计算每列的最大宽度（包括所有行）"""
    if not all_rows:
        return []
    col_count = len(all_rows[0])
    widths = [0] * col_count
    for row in all_rows:
        for i in range(col_count):
            col_len = len(str(row[i]))
            if col_len > widths[i]:
                widths[i] = col_len
    return widths

def format_row(row, widths):
    """按列宽格式化行，每列左对齐+4空格分隔"""
    formatted = []
    for i, (col, width) in enumerate(zip(row, widths)):
        # 左对齐填充到最大宽度，最后一列不加额外空格（避免注释被截断）
        if i == len(widths) - 1:
            formatted.append(f"{col}")
        else:
            # 填充到最大宽度后，加4个空格分隔列
            formatted.append(f"{col.ljust(width)}    ")
    return ''.join(formatted)

def main():
    scenes = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
    table_file = "result_summary.txt"
    
    # 获取编号和注释
    current_num = load_counter()
    method_name = f"dust3r_{current_num}"
    save_counter(current_num + 1)
    comment = input("请输入本次记录的注释：").strip()
    
    # 构建新行数据
    scene_data = [extract_metric(scene) for scene in scenes]
    new_row = [method_name] + scene_data + [comment]
    
    # 读取历史数据并合并
    all_rows = []
    if os.path.exists(table_file):
        with open(table_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # 按4个以上空格分割（兼容旧格式），去除空字符串
                row = [col.strip() for col in line.strip('\n').split() if col.strip()]
                if row:
                    all_rows.append(row)
        # 检查表头是否匹配（防止列数变化）
        if len(all_rows[0]) != len(new_row):
            print("警告：表头列数与新数据不匹配，已重置表格格式")
            all_rows = []  # 重置表格
    
    # 若无历史数据，添加表头
    if not all_rows:
        header = ["方法"] + scenes + ["注释"]
        all_rows.append(header)
    
    # 添加新行
    all_rows.append(new_row)
    
    # 计算所有列的最大宽度
    widths = get_max_column_widths(all_rows)
    
    # 格式化所有行
    formatted_lines = [format_row(row, widths) + '\n' for row in all_rows]
    
    # 写入文件
    with open(table_file, 'w', encoding='utf-8') as f:
        f.writelines(formatted_lines)
    
    print(f"已记录到 {table_file}，当前编号：{current_num}（已对齐）")

if __name__ == "__main__":
    main()