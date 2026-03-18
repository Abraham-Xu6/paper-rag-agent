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

def build_and_save_faiss(chunks, save_path="data/vector_store"):
    print("正在调用智谱 API 生成向量，并构建 FAISS 本地索引...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 将向量库持久化保存到本地硬盘
    vector_store.save_local(save_path)
    print(f"FAISS 向量库已成功保存至: {save_path}")
    return vector_store

def local_faiss_search(query: str, top_k: int = 3, load_path="data/vector_store"):
    embeddings = get_embeddings()
    # 注意：allow_dangerous_deserialization=True 是加载本地 FAISS 必备的参数
    vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"\n正在检索问题: '{query}'")
    results = vector_store.similarity_search(query, k=top_k)
    return results