#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档问答系统 - 启动脚本
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能文档问答系统 - 启动器")
        self.root.geometry("500x400")
        self.root.configure(bg='#f0f0f0')

        self.create_widgets()

    def create_widgets(self):
        """创建界面组件"""
        # 标题
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=20)

        title_label = tk.Label(title_frame,
                                text="🚀 智能文档问答系统",
                                font=("Arial", 20, "bold"),
                                bg='#f0f0f0',
                                fg='#2c3e50')
        title_label.pack()

        subtitle_label = tk.Label(title_frame,
                                  text="整合记忆管理和动态检索的智能问答系统",
                                  font=("Arial", 12),
                                  bg='#f0f0f0',
                                  fg='#34495e')
        subtitle_label.pack(pady=5)

        # 版本选择
        choice_frame = tk.Frame(self.root, bg='#f0f0f0')
        choice_frame.pack(pady=30)

        choice_label = tk.Label(choice_frame,
                                text="请选择启动版本:",
                                font=("Arial", 14),
                                bg='#f0f0f0')
        choice_label.pack(pady=10)

        # 按钮容器
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(expand=True, fill='both', pady=20)

        # 简洁版GUI按钮
        simple_btn = tk.Button(button_frame,
                              text="🖥️ 简洁版GUI (推荐)",
                              font=("Arial", 12),
                              bg='#3498db',
                              fg='white',
                              width=25,
                              height=3,
                              command=self.launch_simple_gui,
                              relief=tk.RAISED,
                              bd=2)
        simple_btn.pack(pady=10)

        simple_desc = tk.Label(button_frame,
                                 text="专注于功能4的图形界面\n支持文档上传和智能问答",
                                 font=("Arial", 10),
                                 bg='#f0f0f0',
                                 fg='#7f8c8d')
        simple_desc.pack()

        # 增强版GUI按钮
        enhanced_btn = tk.Button(button_frame,
                               text="🎨 增强版GUI",
                               font=("Arial", 12),
                               bg='#27ae60',
                               fg='white',
                               width=25,
                               height=3,
                               command=self.launch_enhanced_gui,
                               relief=tk.RAISED,
                               bd=2)
        enhanced_btn.pack(pady=10)

        enhanced_desc = tk.Label(button_frame,
                                  text="功能更完整的图形界面\n包含统计分析和记忆管理",
                                  font=("Arial", 10),
                                  bg='#f0f0f0',
                                  fg='#7f8c8d')
        enhanced_desc.pack()

        # 命令行按钮
        cmd_btn = tk.Button(button_frame,
                         text="⌨️ 命令行版本",
                         font=("Arial", 12),
                         bg='#f39c12',
                         fg='white',
                         width=25,
                         height=3,
                         command=self.launch_cmd_version,
                         relief=tk.RAISED,
                         bd=2)
        cmd_btn.pack(pady=10)

        cmd_desc = tk.Label(button_frame,
                             text="原始的命令行交互界面",
                             font=("Arial", 10),
                             bg='#f0f0f0',
                             fg='#7f8c8d')
        cmd_desc.pack()

        # 底部信息
        info_frame = tk.Frame(self.root, bg='#f0f0f0')
        info_frame.pack(side='bottom', pady=20)

        info_label = tk.Label(info_frame,
                               text="版本1.0 - 智能文档问答系统",
                               font=("Arial", 9),
                               bg='#f0f0f0',
                               fg='#95a5a6')
        info_label.pack()

    def launch_simple_gui(self):
        """启动简洁版GUI"""
        try:
            self.root.destroy()
            subprocess.run([sys.executable, "智能文档问答GUI_简洁版.py"], check=True)
        except Exception as e:
            messagebox.showerror("启动错误", f"无法启动简洁版GUI:\n{e}")

    def launch_enhanced_gui(self):
        """启动增强版GUI"""
        try:
            self.root.destroy()
            subprocess.run([sys.executable, "智能文档问答GUI_增强版.py"], check=True)
        except Exception as e:
            messagebox.showerror("启动错误", f"无法启动增强版GUI:\n{e}")

    def launch_cmd_version(self):
        """启动命令行版本"""
        try:
            self.root.destroy()
            subprocess.run([sys.executable, "文档问答系统.py"], check=True)
        except Exception as e:
            messagebox.showerror("启动错误", f"无法启动命令行版本:\n{e}")

def main():
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()