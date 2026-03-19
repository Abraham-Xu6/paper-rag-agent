import streamlit as st
import requests
import time

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="全能学术 RAG 助手", page_icon="📚", layout="wide")

# ================= 辅助函数与状态清理 =================
def fetch_kbs():
    try:
        res = requests.get(f"{API_BASE_URL}/list_kbs")
        return res.json().get("kbs", ["default"])
    except:
        return ["default"]

# 【核心新增】：清理聊天记录的回调函数
def clear_chat_history():
    st.session_state.messages = []

# 初始化状态
if "kbs" not in st.session_state:
    st.session_state.kbs = fetch_kbs()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 侧边栏配置区 =================
with st.sidebar:
    st.header("⚙️ 助手控制台")
    
    # 【修改点 1】：绑定 on_change 回调，切换模式自动清理
    ui_mode = st.radio(
        "选择工作模式", 
        options=["📚 基于知识库问答 (RAG)", "💬 全能大模型闲聊"], 
        index=0,
        on_change=clear_chat_history  # <--- 自动触发清理
    )
    backend_mode = "kb" if "知识库" in ui_mode else "general"
    
    # 【修改点 2】：绑定 on_change 回调，切换模型自动清理
    ui_model = st.selectbox(
        "选择推理模型", 
        options=["glm-4-flash", "glm-4", "glm-4-plus"], 
        index=0,
        on_change=clear_chat_history  # <--- 自动触发清理
    )
    
    st.divider()
    
    if backend_mode == "kb":
        st.subheader("📁 知识库管理")
        
        col1, col2 = st.columns([6, 4])
        with col1:
            # 【修改点 3】：绑定 on_change 回调，切换知识库自动清理
            selected_kb = st.selectbox(
                "选择目标知识库", 
                options=st.session_state.kbs, 
                label_visibility="collapsed",
                on_change=clear_chat_history  # <--- 自动触发清理
            )
        with col2:
            if st.button("🗑️ 删除", use_container_width=True, help="永久删除当前选中的知识库及文件"):
                if selected_kb == "default":
                    st.toast("⚠️ 默认知识库不可删除！")
                else:
                    with st.spinner("正在清理数据..."):
                        try:
                            res = requests.delete(f"{API_BASE_URL}/delete_kb/{selected_kb}")
                            if res.status_code == 200:
                                st.success(res.json()["message"])
                                st.session_state.kbs = fetch_kbs()
                                clear_chat_history() # 物理删除后顺便清理屏幕
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"删除失败: {res.json().get('detail')}")
                        except Exception as e:
                            st.error(f"请求异常: {e}")
                            
        with st.expander("➕ 批量上传文档至知识库", expanded=False):
            new_kb_name = st.text_input("新建或指定知识库名称", value=selected_kb)
            uploaded_files = st.file_uploader("支持批量拖拽 PDF 格式", type=["pdf"], accept_multiple_files=True)
            
            if st.button("开始批量上传与向量化", use_container_width=True):
                if uploaded_files and new_kb_name:
                    with st.spinner(f"🚀 正在处理 {len(uploaded_files)} 个文件，这可能需要几十秒，请耐心等待..."):
                        files_payload = [
                            ("files", (file.name, file.getvalue(), "application/pdf")) 
                            for file in uploaded_files
                        ]
                        data = {"kb_name": new_kb_name}
                        try:
                            response = requests.post(f"{API_BASE_URL}/upload", files=files_payload, data=data)
                            if response.status_code == 200:
                                st.success(response.json()["message"])
                                st.session_state.kbs = fetch_kbs()
                                time.sleep(1.5) 
                                st.rerun() 
                            else:
                                st.error(f"上传失败: {response.json().get('detail')}")
                        except Exception as e:
                            st.error(f"请求异常: {e}")
                else:
                    st.warning("请先填写知识库名称并选择至少一个文件！")

    st.divider()
    # 【修改点 4】：新增手动“清空聊天记录”按钮
    if st.button("✨ 清空聊天记录", use_container_width=True):
        clear_chat_history()
        st.rerun() # 点击后强制刷新页面以清空屏幕

# ================= 主聊天区域 =================
st.title("📚 全能学术 RAG 研判助手")
status_text = f"**模式**：{ui_mode} | **模型**：`{ui_model}`"
if backend_mode == "kb":
    status_text += f" | **当前知识库**：`{selected_kb}`"
st.markdown(status_text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 查看底层检索来源"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")

if prompt := st.chat_input("请提问，例如：总结核心贡献？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner(f"🧠 `{ui_model}` 正在处理中..."):
            try:
                payload = {
                    "query": prompt,
                    "mode": backend_mode,
                    "model_name": ui_model,
                    "kb_name": selected_kb if backend_mode == "kb" else "default"
                }
                response = requests.post(f"{API_BASE_URL}/chat", json=payload)
                
                if response.status_code != 200:
                    st.error(f"❌ 报错: {response.json().get('detail')}")
                else:
                    result = response.json()
                    answer = result.get("answer", "抱歉，获取回答失败。")
                    sources = result.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("🔍 查看底层检索来源"):
                            for source in sources:
                                st.caption(f"📄 {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "sources": sources
                    })
                    
            except Exception as e:
                st.error(f"❌ 通信失败: 请确保后端 server.py 已启动！")
