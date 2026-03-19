import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from core.embedding_manager import get_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv() 

# 调优的学术级 Prompt
QA_TEMPLATE = """你是一个精通深度学习与人类活动识别（HAR）领域的资深学术研判助手。
你的任务是根据下方【已知信息】来严谨地回答问题。

请严格遵守以下核心规则：
1. 提取文献名称：当用户询问“哪篇文献”、“文献名字”或“出处”时，务必从【已知信息】中每个片段开头的“出处 [X]”中提取出【文件名】告知用户。
2. 拒绝幻觉：在回答具体参数或性能指标时，必须严格基于给定的检索片段。
3. 信息不足处理：如果已知信息中确实找不到答案，请以自然流畅的语气回答“抱歉，检索到的文献片段中未提及该信息”，绝不可自行编造。答案请使用中文。

【已知信息】
{context}

【问题】
{question}

请回答："""

QA_PROMPT = PromptTemplate(template=QA_TEMPLATE, input_variables=["context", "question"])

def get_llm(model_name="glm-4"):
    """初始化智谱 AI 大模型"""
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("环境变量 ZHIPUAI_API_KEY 未设置！请在 .env 文件中配置")
    
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,           
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        temperature=0.7,
        max_tokens=2048
    )

def get_qa_chain(model_name="glm-4-flash", kb_name="default"):
    # 动态拼接这个知识库的本地路径
    vector_store_path = os.path.join("data", "vector_store", kb_name)
    llm = get_llm(model_name)
    embeddings = get_embeddings()
    
    # 增加容错机制：如果没找到对应知识库，抛出明确异常
    if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        raise FileNotFoundError(f"知识库 '{kb_name}' 不存在或尚未上传文件！")
    
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    # 将向量库转换为检索器，设置返回 Top-6 最相关的文档块
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    # 配置内存模块，用于记录多轮历史对话
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer" # 指定哪个输出作为答案存入记忆
    )

    # 组装 ConversationalRetrievalChain（对话检索链）
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True # 方便后续排错查看究竟检索到了哪些原文
    )
    
    return chain

# 备用：一个不依赖检索、纯靠大模型自身知识的对话链（用于对比测试）
def get_general_chain(model_name="glm-4-flash"):
    llm = get_llm(model_name)
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    # 使用基础的 ConversationChain，纯依靠大模型自身知识
    return ConversationChain(llm=llm, memory=memory)
