import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from core.agent_chain import get_qa_chain, get_general_chain

app = FastAPI(title="全能学术 RAG 智能体 API")

# 【核心架构】：定义一个内存字典，用于缓存不同模式和模型的对话链
# 避免每次请求都重新加载向量库，压榨 CPU 性能
chains_cache = {"kb": {}, "general": {}}

@app.on_event("startup")
def startup_event():
    print("🚀 正在启动服务，预加载默认模型与知识库...")
    # 【修改这里】：明确指定 model_name 参数
    chains_cache["kb"]["glm-4-flash"] = get_qa_chain(model_name="glm-4-flash")
    chains_cache["general"]["glm-4-flash"] = get_general_chain(model_name="glm-4-flash")
    print("✅ 服务启动完毕！API 接口已就绪。")

# 【修改点】：扩展 JSON 协议，增加 mode 和 model_name
class ChatRequest(BaseModel):
    query: str
    mode: str = "kb"          # "kb" (知识库) 或 "general" (闲聊)
    model_name: str = "glm-4-flash"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    mode = request.mode
    model_name = request.model_name
    query = request.query

    # 动态加载机制：如果用户选了新模型，就在缓存里新实例化一个
    if model_name not in chains_cache[mode]:
        print(f"🔄 动态加载新链路: {mode} 模式 - {model_name} 模型...")
        if mode == "kb":
            # 【修改这里】：明确指定 model_name 参数
            chains_cache[mode][model_name] = get_qa_chain(model_name=model_name)
        else:
            # 【修改这里】：明确指定 model_name 参数
            chains_cache[mode][model_name] = get_general_chain(model_name=model_name)

    chain = chains_cache[mode][model_name]
    print(f"📨 收到请求: [{mode} | {model_name}] {query}")
    
    # 智能路由执行逻辑
    if mode == "kb":
        res = chain.invoke({"question": query})
        sources = list(set([doc.metadata.get('source', '未知出处') for doc in res['source_documents']]))
        return {"answer": res['answer'], "sources": sources}
    else:
        # General 模式直接闲聊，输入参数是 input，输出参数是 response，无出处
        res = chain.invoke({"input": query})
        return {"answer": res['response'], "sources": []}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
