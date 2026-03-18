import os
from dotenv import load_dotenv
from core.document_processor import process_pdf
from core.embedding_manager import build_and_save_faiss, local_faiss_search

# 加载 .env 文件中的 ZHIPUAI_API_KEY
load_dotenv()

def main():
    pdf_path = "data/docs/Diff_HAR__Semantically_Guided_Diffusion_Generation_for_Generalized_Zero_Shot_Human_Activity_Recognition.pdf" # 请确保你的 PDF 名字和这里一致
    
    # 1. 解析与切块
    chunks = process_pdf(pdf_path)
    
    # 2. 向量化并构建 FAISS 库
    build_and_save_faiss(chunks)
    
    # 3. 测试检索效果 (Top-3 召回)
    test_query = "Diff-HAR 框架在 PAMAP2 数据集上的 H-mean 准确率是多少？"
    retrieved_docs = local_faiss_search(test_query, top_k=3)
    
    print("\n" + "="*40)
    print("【Top-K 召回结果展示】")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- 第 {i+1} 个召回片段 ---")
        print(doc.page_content) # 打印包含“出处”的文本内容
        print("-"*30)

if __name__ == "__main__":
    main()