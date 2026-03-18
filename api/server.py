import os
import sys
# 确保能够导入 core 目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from core.agent_chain import get_qa_chain

# 初始化 FastAPI 应用
app = FastAPI(
    title="学术文献 RAG 智能体 API",
    description="基于本地 FAISS 和智谱 GLM-4 的学术研判接口",
    version="1.0.0"
)

# 声明一个全局变量来存储对话链（避免每次请求都重新加载向量库，压榨 Ryzen 5 5600 的性能）
qa_chain = None

@app.on_event("startup")
def startup_event():
    global qa_chain
    print("🚀 正在启动服务，加载大模型与本地知识库...")
    qa_chain = get_qa_chain()
    print("✅ 服务启动完毕！API 接口已就绪。")

# 定义前端传过来的 JSON 数据格式
class ChatRequest(BaseModel):
    query: str

# 定义 POST 接口
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not qa_chain:
        return {"error": "服务未初始化完成，请稍后再试"}
    
    print(f"收到请求: {request.query}")
    
    # 核心：调用我们在 core 目录写好的逻辑
    res = qa_chain.invoke({"question": request.query})
    
    # 提取并去重底层的来源文件路径 (PyMuPDFLoader 会自动把路径存在 metadata 里)
    sources = list(set([doc.metadata.get('source', '未知出处') for doc in res['source_documents']]))
    
    # 返回标准的 JSON 格式给前端
    return {
        "answer": res['answer'],
        "sources": sources
    }

if __name__ == "__main__":
    # 启动 Uvicorn 服务器，监听本地 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)