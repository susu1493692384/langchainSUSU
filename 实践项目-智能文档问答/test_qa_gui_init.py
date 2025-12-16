#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试GUI应用初始化
只初始化GUI而不运行主循环
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gui_init():
    """测试GUI初始化"""
    try:
        print("正在导入CustomTkinter...")
        import customtkinter as ctk
        print("✅ CustomTkinter导入成功")

        print("正在设置外观...")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        print("✅ 外观设置成功")

        print("正在创建主窗口...")
        app = ctk.CTk()
        app.title("测试窗口")
        app.geometry("400x300")
        print("✅ 主窗口创建成功")

        print("正在导入问答系统...")
        from 文档问答系统 import IntelligentDocumentQA
        print("✅ 问答系统导入成功")

        print("正在测试问答系统初始化...")
        # 检查是否有必要的API配置
        import os
        if os.getenv("GLM_API_KEY") and os.getenv("GLM_BASE_URL"):
            qa_system = IntelligentDocumentQA()
            print("✅ 问答系统初始化成功")
        else:
            print("⚠️ 未找到API配置，跳过问答系统初始化")

        print("正在创建测试组件...")
        # 创建测试组件
        frame = ctk.CTkFrame(app)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        label = ctk.CTkLabel(frame, text="GUI测试成功！", font=ctk.CTkFont(size=16))
        label.pack(pady=10)

        button = ctk.CTkButton(frame, text="关闭测试", command=app.destroy)
        button.pack(pady=10)

        print("✅ 所有测试组件创建成功")
        print("GUI初始化测试完成！")
        print("请关闭测试窗口继续...")

        # 显示窗口但不运行mainloop（会阻塞）
        app.after(100, app.deiconify)  # 确保窗口可见
        app.mainloop()

    except Exception as e:
        print(f"❌ GUI初始化测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui_init()