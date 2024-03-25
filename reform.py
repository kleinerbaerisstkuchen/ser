# import json
# import time
# # 读取整个文件内容
# t1 = time.time()
# with open('/home/hnaxiong/data/youtube_vis/labels/train.json', 'r') as file:
#     json_content = file.read()
# t2 = time.time()
# print(f"读取时间为{t2-t1}s")
# # 解析 JSON 字符串
# try:
#     parsed_json = json.loads(json_content)
# except json.JSONDecodeError as e:
#     print("JSON 解析错误:", e)
#     exit(1)

# # 格式化 JSON 字符串（添加缩进和换行）
# formatted_json = json.dumps(parsed_json, indent=4)
# t3 = time.time()
# print(f"处理时间为{t3-t2}s")
# # 将格式化后的 JSON 写入新文件中
# with open('/home/hnaxiong/ser/formated_train.json', 'w') as file:
#     file.write(formatted_json)
# t4 = time.time()
# print("JSON 文件已格式化并保存为 'formatted_json_file.json'")
# print(f"写入时间为{t4-t3}s")



# # 计算 JSON 文件中的行数
# def count_lines_json(file_path):
#     with open(file_path, 'r') as file:
#         line_count = sum(1 for line in file)
#     return line_count

# # JSON 文件路径
# json_file_path = '/home/hnaxiong/Downloads/train.json'

# # 调用函数并打印行数
# line_count = count_lines_json(json_file_path)
# print("JSON 文件共有 {} 行".format(line_count))





# import json

# # 处理大型 JSON 文件的部分内容
# def process_partial_json(input_file, output_file, num_chars):
#     with open(input_file, 'r') as infile:
#         # 读取文件的部分内容（前 num_chars 个字符）
#         json_data = infile.read(num_chars)

#     # 解析 JSON 数据
#     try:
#         json_obj = json.loads(json_data)
#     except json.JSONDecodeError as e:
#         print("JSON 解析错误:", e)
#         return

#     # 将解析后的 JSON 数据以正常格式写入到输出文件中
#     with open(output_file, 'w') as outfile:
#         json.dump(json_obj, outfile, indent=4)

#     print("部分 JSON 文件已处理完成")

# # 处理大型 JSON 文件的部分内容
# process_partial_json('/home/hnaxiong/Downloads/train.json', 'partial_reformed_json_file.json', 9000)


import json

# 读取 JSON 文件
with open('/home/hnaxiong/Downloads/train.json', 'r') as f:
    data = json.load(f)

# 创建一个新的字典来保存提取的数据
extracted_data = {
    'info': data['info'],
    'licenses': data['licenses'],
    'videos': data['videos'][:10],  # 提取前10个videos
    'categories': data['categories'],
    'annotations': data['annotations'][:100]  # 提取前100个annotations
}

# 将提取的数据保存到新文件中
with open('/home/hnaxiong/Downloads/extracted_data.json', 'w') as f:
    json.dump(extracted_data, f, indent=4)
