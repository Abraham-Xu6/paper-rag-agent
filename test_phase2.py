from dotenv import load_dotenv
from core.agent_chain import get_qa_chain

# 加载环境变量中的 API Key
load_dotenv()

def main():
    print("正在初始化 RAG 智能体对话链路...")
    qa_chain = get_qa_chain()
    
    print("\n" + "="*50)
    print("🎓 学术文献研判助手已就绪！")
    print("你可以连续问我问题了（输入 'quit' 退出）。")
    print("="*50)
    
    while True:
        query = input("\n🧐 你的问题：")
        if query.lower() in ['quit', 'exit']:
            print("再见！")
            break
            
        if not query.strip():
            continue
            
        # 向对话链发起请求
        # 因为 memory 内部自动维护了上下文，所以每次只需要传当前 question 即可
        res = qa_chain.invoke({"question": query})
        
        print(f"\n🤖 助手回答：\n{res['answer']}")
        
        # 可选：打印检索到的出处来源（排错神器）
        print("\n[底层检索来源调试信息]:")
        for i, doc in enumerate(res['source_documents']):
            # 仅打印开头前 40 个字符以便确认是否包含了文件名
            print(f"  来源 {i+1}: {doc.page_content[:40].replace(chr(10), '')}...") 

if __name__ == "__main__":
    main()