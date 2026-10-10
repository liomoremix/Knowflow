# test/fastapi/demo.py

from fastapi import FastAPI

# 创建一个 FastAPI 应用实例
app = FastAPI()

@app.get("/", summary="第一个 测试")
async def read_root():
    return {"Hello": "World"}



# test/fastapi/demo.py

# 访问 http://127.0.0.1:8000/items/5?q=somequery
# item_id: 路径参数 (自动转为 int)
# q: 查询参数 (可选，默认 None)
@app.get("/items/{item_id}", summary="获取指定参数")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


# 接收? skip=? & limit = ?
@app.get("/items", summary="分页")
async def read_item(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}



from pydantic import BaseModel

# 定义数据模型
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

# POST 请求接收 JSON 数据
@app.post("/items/", summary="类型检查")
async def create_item(item: Item):
    # item 已经是验证过的 Item 对象
    # 如果客户端传来的 price 是字符串 "abc"，FastAPI 会自动报错
    return {"name": item.name, "price": item.price, "is_offer": item.is_offer}


from fastapi.responses import JSONResponse

@app.get("/api/user")
async def get_user():
    # 等价于直接 return {"name": "张三", "age": 20}（FastAPI 自动转 JSONResponse）
    return JSONResponse(
        content={"name": "张三", "age": 20},
        status_code=200,  # 可选，默认 200
        headers={"X-Custom-Header": "custom-value"}  # 可选，自定义响应头
    )





# test/fastapi/demo.py

from fastapi.responses import FileResponse

@app.get("/download/excel")
async def download_excel():
    excel_path = "D:/test.xls"
    # 返回文件并指定下载文件名
    return FileResponse(
        path=excel_path,
        filename="月度报表.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
