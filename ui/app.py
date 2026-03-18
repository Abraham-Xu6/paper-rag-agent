import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

# 采用 wide 布局，给侧边栏腾出空间
st.set_page_config(page_title="全能学术 RAG 助手", page_icon="📚", layout="wide")

# ================= 侧边栏配置区 =================
with st.sidebar:
    st.header("⚙️ 助手控制台")
    
    # 模式选择器
    ui_mode = st.radio(
        "选择工作模式",
        options=["📚 基于知识库问答 (RAG)", "💬 全能大模型闲聊"],
        index=0
    )
    
    # 模型选择器
    ui_model = st.selectbox(
        "选择推理模型",
        options=["glm-4-flash", "glm-4", "glm-4-plus"],
        index=0,
        help="flash 速度极快且免费；glm-4 逻辑推理与长文本能力更强。"
    )
    
    st.divider()
    st.caption("✨ 提示：切换模式或模型后，系统会在后端自动智能路由。")

    # 将前端的中文选项映射为后端的英文代码
    backend_mode = "kb" if "知识库" in ui_mode else "general"

# ================= 主聊天区域 =================
st.title("📚 全能学术 RAG 研判助手")
st.markdown(f"**当前状态**：{ui_mode}  |  **驱动核心**：`{ui_model}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 查看底层检索来源"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")

if prompt := st.chat_input("请提问，例如：什么是 Diffusion 模型？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner(f"🧠 `{ui_model}` 正在思考中..."):
            try:
                # 【修改点】：将前端设置好的模式和模型参数发送给后端
                payload = {
                    "query": prompt,
                    "mode": backend_mode,
                    "model_name": ui_model
                }
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() 
                
                result = response.json()
                answer = result.get("answer", "抱歉，获取回答失败。")
                sources = result.get("sources", [])
                
                st.markdown(answer)
                
                if sources:
                    with st.expander("🔍 查看底层检索来源"):
                        for source in sources:
                            st.caption(f"📄 {source}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"❌ 通信失败: 请确保后端 server.py 已启动！错误详情: {str(e)}")
