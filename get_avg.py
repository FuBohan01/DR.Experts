import re
import os

def extract_values_from_line(line, plcc_pattern, srcc_pattern):
    plcc_matches = plcc_pattern.findall(line)
    srcc_matches = srcc_pattern.findall(line)
    return [float(match) for match in plcc_matches], [float(match) for match in srcc_matches]

def calculate_mean(values):
    return sum(values) / len(values) if values else 0

def calculate_median(values):
    if not values:
        return 0
    sorted_values = sorted(values)
    mid_index = len(sorted_values) // 2
    return (sorted_values[mid_index] + sorted_values[~mid_index]) / 2 if len(values) % 2 else sorted_values[mid_index]

base_path = "/home/fubohan/Code/DIQA/results/base_size/full_livec_deit[base]_daclip_v2_diffv3_DIN"
plcc_values = []
srcc_values = []
indexes = []  # 记录每个srcc和plcc的索引

plcc_pattern = re.compile(r"Max PLCC:\s*(\d+\.\d+)")
srcc_pattern = re.compile(r"Max SRCC:\s*(\d+\.\d+)")

for i in range(20):
    log_file_path = os.path.join(base_path, str(i), "log_rank0.txt")
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) >= 2:
                second_to_last_line = lines[-2]
                plcc, srcc = extract_values_from_line(second_to_last_line, plcc_pattern, srcc_pattern)
                for p, s in zip(plcc, srcc):
                    plcc_values.append(p)
                    srcc_values.append(s)
                    indexes.append(i)  # 记录索引
            else:
                print(f"Not enough lines in log file: {log_file_path}")
    else:
        print(f"Log file not found: {log_file_path}")

# 根据srcc排序并获取前10个索引
sorted_srcc_indexes = sorted(range(len(srcc_values)), key=lambda k: srcc_values[k], reverse=True)[:10]

# 使用索引获取对应的plcc值
top_10_plcc_values = [plcc_values[i] for i in sorted_srcc_indexes]
top_10_srcc_values = [srcc_values[i] for i in sorted_srcc_indexes]

print(srcc_values)

# 计算平均值和中位数
mean_srcc = calculate_mean(top_10_srcc_values)
mean_plcc = calculate_mean(top_10_plcc_values)
median_srcc = calculate_median(top_10_srcc_values)
median_plcc = calculate_median(top_10_plcc_values)


print(f"Mean Max SRCC: {mean_srcc:.6f}")
print(f"Mean Max PLCC: {mean_plcc:.6f}")
print(f"Median Max SRCC: {median_srcc:.6f}")
print(f"Median Max PLCC: {median_plcc:.6f}")