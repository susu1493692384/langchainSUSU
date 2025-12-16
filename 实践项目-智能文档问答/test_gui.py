#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的GUI测试脚本
用于测试CustomTkinter是否正常工作
"""

import customtkinter as ctk
import tkinter as tk

def test_gui():
    """测试GUI基本功能"""
    print("正在测试CustomTkinter GUI...")

    # 设置外观
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    # 创建窗口
    app = ctk.CTk()
    app.title("GUI测试")
    app.geometry("400x300")

    # 添加标签
    label = ctk.CTkLabel(
        app,
        text="CustomTkinter 测试成功！",
        font=ctk.CTkFont(size=20, weight="bold")
    )
    label.pack(pady=20)

    # 添加按钮
    def button_click():
        print("按钮被点击了！")
        label.configure(text="按钮点击测试成功！")

    button = ctk.CTkButton(
        app,
        text="测试按钮",
        command=button_click
    )
    button.pack(pady=10)

    # 添加输入框
    entry = ctk.CTkEntry(app, placeholder_text="输入测试文本")
    entry.pack(pady=10)

    def show_input():
        text = entry.get()
        print(f"输入的文本: {text}")
        if text:
            label.configure(text=f"输入成功: {text}")
        else:
            label.configure(text="请输入文本")

    submit_btn = ctk.CTkButton(
        app,
        text="提交",
        command=show_input
    )
    submit_btn.pack(pady=5)

    print("GUI界面已创建，请关闭窗口继续...")

    # 运行应用
    app.mainloop()

    print("GUI测试完成！")

if __name__ == "__main__":
    try:
        test_gui()
        print("✅ CustomTkinter GUI测试成功")
    except Exception as e:
        print(f"❌ GUI测试失败: {e}")