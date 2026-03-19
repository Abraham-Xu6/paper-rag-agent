import subprocess
import sys
import time
import atexit

def main():
    print("="*50)
    print("🚀 正在启动全能学术 RAG 智能体系统...")
    print("="*50)
    
    # 记录拉起的进程
    processes = []

    # 1. 启动后端 FastAPI 服务
    print("\n[1/2] 正在启动后端服务 (FastAPI)...")
    # sys.executable 会自动获取你当前 Conda 虚拟环境的 Python 解释器路径
    backend_process = subprocess.Popen([sys.executable, "api/server.py"])
    processes.append(backend_process)
    
    # 稍微等待 3 秒，确保后端的 8000 端口已经绑定完毕，再启动前端
    print("⏳ 等待后端初始化...")
    time.sleep(3)
    
    # 2. 启动前端 Streamlit 服务
    print("\n[2/2] 正在启动前端服务 (Streamlit)...")
    frontend_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/app.py"])
    processes.append(frontend_process)
    
    print("\n" + "="*50)
    print("✅ 系统已全部启动！")
    print("👉 请在浏览器访问前端页面 (通常是 http://localhost:8501)")
    print("🛑 按下 Ctrl + C 可以同时关闭前后端服务")
    print("="*50 + "\n")

    # 3. 注册清理函数：当主程序退出时，自动终结所有子进程
    def cleanup():
        print("\n🛑 接收到退出信号，正在关闭所有服务...")
        for p in processes:
            p.terminate()
            p.wait() # 等待进程彻底死掉
        print("✅ 服务已完全关闭，端口已释放。")
        
    atexit.register(cleanup)
    
    # 4. 保持主进程运行，直到用户按下 Ctrl+C
    try:
        # 让主进程阻塞在这里，监听子进程的状态
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        # 捕获用户的 Ctrl+C 操作，什么都不用写，atexit 会自动触发 cleanup
        pass

if __name__ == "__main__":
    main()