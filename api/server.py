import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uvicorn

from core.agent_chain import get_qa_chain, get_general_chain
from core.document_processor import process_pdf
from core.embedding_manager import build_and_save_faiss

from typing import List 

app = FastAPI(title="全能学术 RAG 智能体 API")

# 缓存结构升级：chains_cache["kb"][kb_name][model_name]
chains_cache = {"kb": {}, "general": {}}

# --- 新增：获取现有知识库列表接口 ---
@app.get("/list_kbs")
async def list_kbs():
    base_dir = os.path.join("data", "vector_store")
    if not os.path.exists(base_dir):
        return {"kbs": []}
    kbs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return {"kbs": kbs}

# --- 修改：多文件并发上传与向量化接口 ---
@app.post("/upload")
async def upload_document(files: List[UploadFile] = File(...), kb_name: str = Form(...)): # 注意这里改成了 files: List[UploadFile]
    # 1. 准备目标知识库的物理目录
    doc_dir = os.path.join("data", "docs", kb_name)
    os.makedirs(doc_dir, exist_ok=True)
    
    all_chunks = []
    saved_filenames = []
    
    try:
        # 2. 遍历处理每一个上传的文件
        for file in files:
            if not file.filename.endswith(".pdf"):
                continue # 遇到非 PDF 文件跳过，防止程序崩溃
                
            file_path = os.path.join(doc_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # 解析单个 PDF 并将切分出的文本块追加到总列表中
            chunks = process_pdf(file_path)
            all_chunks.extend(chunks)
            saved_filenames.append(file.filename)
            
        if not all_chunks:
            raise HTTPException(status_code=400, detail="没有检测到有效的 PDF 文件内容")

        # 3. 将所有文件的文本块【一次性】交给 FAISS 处理（新增或追加）
        build_and_save_faiss(all_chunks, kb_name)
        
        # 4. 清理旧缓存
        if kb_name in chains_cache.get("kb", {}):
            del chains_cache["kb"][kb_name]
            
        return {
            "message": f"成功将 {len(saved_filenames)} 个文件加入知识库 [{kb_name}]！共解析 {len(all_chunks)} 个片段。"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- 新增：删除知识库接口 ---
@app.delete("/delete_kb/{kb_name}")
async def delete_knowledge_base(kb_name: str):
    # 【安全防御】：禁止删除默认兜底的知识库
    if kb_name == "default":
        raise HTTPException(status_code=400, detail="系统默认知识库 (default) 保留，不可删除。")

    # 物理路径定位
    vector_dir = os.path.join("data", "vector_store", kb_name)
    docs_dir = os.path.join("data", "docs", kb_name)

    try:
        # 1. 删除硬盘上的物理文件（向量库与原始PDF）
        if os.path.exists(vector_dir):
            shutil.rmtree(vector_dir)
        if os.path.exists(docs_dir):
            shutil.rmtree(docs_dir)

        # 2. 清理内存中的链路缓存（防止出现内存泄漏和幽灵检索）
        if "kb" in chains_cache and kb_name in chains_cache["kb"]:
            del chains_cache["kb"][kb_name]
            print(f"🧹 已清理知识库 [{kb_name}] 的内存缓存")

        return {"message": f"知识库 [{kb_name}] 及相关数据已彻底删除！"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除过程中发生错误: {str(e)}")

# --- 修改：聊天接口适配多知识库 ---
class ChatRequest(BaseModel):
    query: str
    mode: str = "kb"          
    model_name: str = "glm-4-flash"
    kb_name: str = "default"  # 新增 kb_name 参数

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    mode = request.mode
    model_name = request.model_name
    kb_name = request.kb_name
    query = request.query

    try:
        if mode == "general":
            if model_name not in chains_cache["general"]:
                chains_cache["general"][model_name] = get_general_chain(model_name=model_name)
            chain = chains_cache["general"][model_name]
            res = chain.invoke({"input": query})
            return {"answer": res['response'], "sources": []}
        
        elif mode == "kb":
            # 初始化特定 kb_name 的缓存字典
            if kb_name not in chains_cache["kb"]:
                chains_cache["kb"][kb_name] = {}
                
            if model_name not in chains_cache["kb"][kb_name]:
                print(f"🔄 动态加载新链路: 知识库[{kb_name}] - {model_name} 模型...")
                chains_cache["kb"][kb_name][model_name] = get_qa_chain(model_name=model_name, kb_name=kb_name)
                
            chain = chains_cache["kb"][kb_name][model_name]
            res = chain.invoke({"question": query})
            sources = list(set([doc.metadata.get('source', '未知出处') for doc in res['source_documents']]))
            return {"answer": res['answer'], "sources": sources}
            
    except FileNotFoundError as e:
         raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
