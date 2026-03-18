import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path: str):
    print(f"正在加载文献: {file_path} ...")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    print("正在执行精准切分(Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""] ,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(docs)

    # 元数据显式注入 (幻觉抑制)
    filename = os.path.basename(file_path)
    for chunk in chunks:
        # 强制在每个文本块的开头拼接出处信息，方便后续大模型精准溯源
        chunk.page_content = f"出处 [{filename}] : \n{chunk.page_content}"

    print(f"切分完成！共生成 {len(chunks)} 个文本块(Chunks)。")
    return chunks