# test/bge_m3测试向量转换.py
from utils.embedding_utils import get_bge_m3_ef

model = get_bge_m3_ef()
result = model.encode_documents(["测试","test"])

print(result)
# 数据结构：
# 稠密向量：这是一个包含两个 array 的列表。每个 array 长度为 1024，数据类型是 float16（半精度）。
# 稀疏向量：这是一个 Compressed Sparse Row sparse array（压缩稀疏行矩阵），形状是 (2, 250002)，250002 代表模型词汇表的大小（大约有 25 万个可能的词），2表示当前是2个词的向量