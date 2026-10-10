# processor/import_processor/nodes/node_entry.py

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.state import ImportGraphState
from processor.import_processor.exceptions import ValidationError, StateFieldError, FileProcessingError
import logging
import json
from pathlib import Path


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        """
        1.接收状态: 获取 `import_file_path`。
        2.判断类型: 检查文件后缀是 `.pdf` 还是 `.md`。
        3.设置标记: 更新 state 中的 `is_pdf_read_enabled/pdf_path` 或 `is_md_read_enabled/md_path`，供主图路由使用。
        4.提取标题: 从文件名中提取 `file_title`，后续作为元数据。
        :param state: `import_file_path`
        :return: `is_pdf_read_enabled/pdf_path` 或 `is_md_read_enabled/md_path` 、`file_title`
        """

        # 1. 从state中获取文件
        import_file_path = state.get("import_file_path")
        # import_file_path = state["import_file_path"] #这样写的话字段不存在直接报错
        # 判断路劲是否为空
        if not import_file_path:
            raise StateFieldError(field_name="import_file_path", expected_type=str)

        # 2. 转换Path为标准化对象
        import_file_path_obj = Path(import_file_path)
        # 判断文件是否存在
        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件{import_file_path_obj.name}不存在")

        # 3. 检查文件后缀
        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
            # 直接导入 .md 时无人加载正文,此处兜底读入,避免下游 md_content 缺失
            if not state.get("md_content"):
                state["md_content"] = import_file_path_obj.read_text(encoding="utf-8")
        else:
            raise ValidationError(message=f"该文件的后缀格式{import_file_path_obj.suffix}不支持")

        # 4. 获取上传文件标题，更新到 state 中
        state["file_title"] = import_file_path_obj.stem

        return state



if __name__ == "__main__":
    setup_logging()

    init_state = {"import_file_path": r"D:\doc\hak180产品安全手册.pdf"}
    node_entry = NodeEntry()
    result = node_entry(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))