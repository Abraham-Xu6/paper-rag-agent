import streamlit as st
import requests

# FastAPI 后端地址
API_URL = "http://127.0.0.1:8000/chat"

# 设置网页标题和布局
st.set_page_config(page_title="学术文献 RAG 助手", page_icon="📕", layout="centered")

st.title("📕 学术文献 RAG 研判助手")
st.markdown("基于本地 FAISS + 智谱 GLM-4 的专属科研引擎")

# 初始化聊天记录（Session State 状态保持）
# 这是网页刷新后聊天气泡不丢失的核心秘密
if "messages" not in st.session_state:
    st.session_state.messages = []

# 遍历并展示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果有来源信息，使用折叠面板优雅地展示
        if "sources" in message and message["sources"]:
            with st.expander("🔍 查看底层检索来源"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")

# 获取底部输入框的用户提问
if prompt := st.chat_input("请提问，例如：Diff-HAR 框架为了解决什么痛点？"):
    # 1. 立即把用户的问题显示在界面上
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. 把用户的问题存入历史记录
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. 向 FastAPI 发起请求并展示回答
    with st.chat_message("assistant"):
        with st.spinner("🧠 正在检索本地知识库并思考中..."):
            try:
                # 使用 requests 库向后端 API 发送 JSON 数据
                response = requests.post(API_URL, json={"query": prompt})
                response.raise_for_status() # 检查 HTTP 状态码
                
                # 解析返回的 JSON 数据
                result = response.json()
                answer = result.get("answer", "抱歉，获取回答失败。")
                sources = result.get("sources", [])
                
                # 打字机效果展示回答
                st.markdown(answer)
                
                # 优雅地展示溯源文件
                if sources:
                    with st.expander("🔍 查看底层检索来源"):
                        for source in sources:
                            st.caption(f"📄 {source}")
                
                # 把助手的回答存入历史记录，为下一轮对话做准备
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到后端服务！请确保 `python api/server.py` 已在另一个终端中启动。")
            except Exception as e:
                st.error(f"❌ 发生未知错误: {str(e)}")