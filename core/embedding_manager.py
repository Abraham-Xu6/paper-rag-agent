import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings():
    # 利用智谱兼容 OpenAI 接口的特性进行底层封装
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("ZHIPUAI_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        model="embedding-3",
        chunk_size=64, # 设置每批处理的文本数量,智谱API对批量嵌入请求有限制，一次最多只能处理 64 条文本
    )

def build_and_save_faiss(chunks, kb_name="default"):
    """
    支持多知识库隔离与追加合并的核心逻辑
    """
    save_path = os.path.join("data", "vector_store", kb_name)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"正在调用智谱 API 生成向量，目标知识库: {kb_name} ...")
    embeddings = get_embeddings()
    
    # 【核心逻辑】：判断知识库是否已存在
    if os.path.exists(os.path.join(save_path, "index.faiss")):
        print("检测到已有知识库，正在执行向量追加(Merge)...")
        # 1. 加载老库
        vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        # 2. 将新切分的 chunks 追加进老库
        vector_store.add_documents(chunks)
    else:
        print("新建知识库，正在执行全量向量化...")
        # 新建向量库
        vector_store = FAISS.from_documents(chunks, embeddings)
        
    # 保存/覆盖本地文件
    vector_store.save_local(save_path)
    print(f"FAISS 向量库 [{kb_name}] 更新成功！")
    return vector_store

def local_faiss_search(query: str, top_k: int = 3, load_path="data/vector_store"):
    embeddings = get_embeddings()
    # 注意：allow_dangerous_deserialization=True 是加载本地 FAISS 必备的参数
    vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"\n正在检索问题: '{query}'")
    results = vector_store.similarity_search(query, k=top_k)
    return results
