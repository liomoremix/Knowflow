# test/03_测试异常处理.py

from processor.import_processor.exceptions import ValidationError

try:
    # 可能失败的操作
    raise ValidationError("数据类型不匹配")
except Exception as e:
    raise ValidationError(
        message="向量写入失败",
        node_name="import_milvus",
        cause=e
    )