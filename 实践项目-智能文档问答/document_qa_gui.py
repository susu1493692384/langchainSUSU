#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档问答系统GUI界面
基于CustomTkinter的现代化卡片式布局界面
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import queue
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime

# 确保导入正确
try:
    from tkinter import filedialog, messagebox
except ImportError:
    # 如果CustomTkinter版本不支持，使用备用方案
    filedialog = None
    messagebox = None

# 设置CustomTkinter外观
ctk.set_appearance_mode("light")  # 可选: "light", "dark", "system"
ctk.set_default_color_theme("green")  # 可选: "blue", "green", "dark-blue"

# 导入现有的问答系统
from 文档问答系统 import IntelligentDocumentQA, create_sample_document_file

class DocumentQAGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("智能文档问答系统")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # 初始化问答系统
        self.qa_system = None
        self.qa_queue = queue.Queue()

        # 界面状态
        self.is_processing = False
        self.current_theme = "light"

        # 设置样式
        self.setup_styles()

        # 创建主界面
        self.create_main_interface()

        # 初始化问答系统（在后台线程中）
        self.initialize_qa_system()

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 定期更新系统信息
        self.update_system_info()

    def setup_styles(self):
        """设置界面样式"""
        # 字体设置
        self.title_font = ctk.CTkFont(family="微软雅黑", size=24, weight="bold")
        self.heading_font = ctk.CTkFont(family="微软雅黑", size=16, weight="bold")
        self.normal_font = ctk.CTkFont(family="微软雅黑", size=12)
        self.small_font = ctk.CTkFont(family="微软雅黑", size=10)

        # 颜色配置
        self.colors = {
            "primary": "#1e6ba8",
            "secondary": "#48cae4",
            "success": "#52b788",
            "warning": "#f77f00",
            "error": "#d62828",
            "background": "#f8f9fa",
            "card_bg": "#ffffff",
            "text_primary": "#212529",
            "text_secondary": "#6c757d"
        }

    def create_main_interface(self):
        """创建主界面布局"""
        # 创建主框架
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 创建顶部标题栏
        self.create_header()

        # 创建内容区域
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True, pady=(10, 0))

        # 使用grid布局组织卡片
        self.content_frame.grid_columnconfigure(1, weight=2)  # 对话区域最宽
        self.content_frame.grid_columnconfigure(0, weight=1)  # 文档管理
        self.content_frame.grid_columnconfigure(2, weight=1)  # 系统信息
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=0)  # 输入区域

        # 创建各个卡片组件
        self.create_document_card()
        self.create_conversation_card()
        self.create_system_info_card()
        self.create_input_card()

        # 创建底部状态栏
        self.create_status_bar()

    def create_header(self):
        """创建顶部标题栏"""
        header_frame = ctk.CTkFrame(self.main_frame, height=60)
        header_frame.pack(fill="x", pady=(0, 10))
        header_frame.pack_propagate(False)

        # 左侧标题
        title_label = ctk.CTkLabel(
            header_frame,
            text="🤖 智能文档问答系统",
            font=self.title_font,
            text_color="#1e6ba8"
        )
        title_label.pack(side="left", padx=20, pady=15)

        # 右侧控制按钮
        controls_frame = ctk.CTkFrame(header_frame)
        controls_frame.pack(side="right", padx=20, pady=10)

        # 主题切换按钮
        self.theme_button = ctk.CTkButton(
            controls_frame,
            text="🌙 暗色主题",
            width=120,
            height=35,
            command=self.toggle_theme
        )
        self.theme_button.pack(side="left", padx=5)

        # 设置按钮
        self.settings_button = ctk.CTkButton(
            controls_frame,
            text="⚙️ 设置",
            width=100,
            height=35,
            command=self.open_settings
        )
        self.settings_button.pack(side="left", padx=5)

        # 帮助按钮
        self.help_button = ctk.CTkButton(
            controls_frame,
            text="❓ 帮助",
            width=100,
            height=35,
            command=self.show_help
        )
        self.help_button.pack(side="left", padx=5)

    def create_document_card(self):
        """创建文档管理卡片"""
        doc_card = ctk.CTkFrame(self.content_frame)
        doc_card.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # 卡片标题
        title_frame = ctk.CTkFrame(doc_card, height=50)
        title_frame.pack(fill="x", padx=10, pady=10)
        title_frame.pack_propagate(False)

        title_label = ctk.CTkLabel(
            title_frame,
            text="📁 文档管理",
            font=self.heading_font,
            text_color="#1e6ba8"
        )
        title_label.pack(side="left", padx=10, pady=10)

        # 上传按钮
        upload_btn = ctk.CTkButton(
            title_frame,
            text="📤 上传文档",
            width=100,
            height=30,
            command=self.upload_document
        )
        upload_btn.pack(side="right", padx=10, pady=10)

        # 文档列表区域
        list_frame = ctk.CTkFrame(doc_card)
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 文档列表标题
        list_title = ctk.CTkLabel(
            list_frame,
            text="已加载文档",
            font=self.normal_font,
            anchor="w"
        )
        list_title.pack(fill="x", padx=10, pady=(10, 5))

        # 文档列表（使用ScrollableFrame）
        self.doc_list_frame = ctk.CTkScrollableFrame(
            list_frame,
            height=300
        )
        self.doc_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 文档预览区域
        preview_frame = ctk.CTkFrame(doc_card)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        preview_title = ctk.CTkLabel(
            preview_frame,
            text="文档预览",
            font=self.normal_font,
            anchor="w"
        )
        preview_title.pack(fill="x", padx=10, pady=(10, 5))

        self.doc_preview = ctk.CTkTextbox(
            preview_frame,
            height=200,
            font=self.small_font
        )
        self.doc_preview.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 文档设置区域
        settings_frame = ctk.CTkFrame(doc_card)
        settings_frame.pack(fill="x", padx=10, pady=(0, 10))

        settings_title = ctk.CTkLabel(
            settings_frame,
            text="文档设置",
            font=self.normal_font,
            anchor="w"
        )
        settings_title.pack(fill="x", padx=10, pady=(10, 5))

        # 分块大小设置
        chunk_frame = ctk.CTkFrame(settings_frame)
        chunk_frame.pack(fill="x", padx=10, pady=5)

        chunk_label = ctk.CTkLabel(
            chunk_frame,
            text="分块大小:",
            width=80
        )
        chunk_label.pack(side="left", padx=5)

        self.chunk_size_var = tk.StringVar(value="300")
        chunk_entry = ctk.CTkEntry(
            chunk_frame,
            textvariable=self.chunk_size_var,
            width=80
        )
        chunk_entry.pack(side="left", padx=5)

        chunk_help = ctk.CTkLabel(
            chunk_frame,
            text="字符数",
            font=self.small_font,
            text_color="gray"
        )
        chunk_help.pack(side="left", padx=5)

    def create_conversation_card(self):
        """创建对话交互卡片"""
        conv_card = ctk.CTkFrame(self.content_frame)
        conv_card.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # 卡片标题
        title_frame = ctk.CTkFrame(conv_card, height=50)
        title_frame.pack(fill="x", padx=10, pady=10)
        title_frame.pack_propagate(False)

        title_label = ctk.CTkLabel(
            title_frame,
            text="💬 智能对话",
            font=self.heading_font,
            text_color="#1e6ba8"
        )
        title_label.pack(side="left", padx=10, pady=10)

        # 控制按钮
        button_frame = ctk.CTkFrame(title_frame)
        button_frame.pack(side="right", padx=10, pady=10)

        clear_btn = ctk.CTkButton(
            button_frame,
            text="🗑️ 清空",
            width=80,
            height=30,
            command=self.clear_conversation
        )
        clear_btn.pack(side="left", padx=5)

        export_btn = ctk.CTkButton(
            button_frame,
            text="💾 导出",
            width=80,
            height=30,
            command=self.export_conversation
        )
        export_btn.pack(side="left", padx=5)

        # 对话历史区域
        self.conversation_frame = ctk.CTkScrollableFrame(
            conv_card,
            height=600
        )
        self.conversation_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 初始欢迎消息
        self.add_welcome_message()

        # 输入区域将在底部单独创建

    def create_system_info_card(self):
        """创建系统信息监控卡片"""
        info_card = ctk.CTkFrame(self.content_frame)
        info_card.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # 卡片标题
        title_frame = ctk.CTkFrame(info_card, height=50)
        title_frame.pack(fill="x", padx=10, pady=10)
        title_frame.pack_propagate(False)

        title_label = ctk.CTkLabel(
            title_frame,
            text="📊 系统监控",
            font=self.heading_font,
            text_color="#1e6ba8"
        )
        title_label.pack(side="left", padx=10, pady=10)

        # 刷新按钮
        refresh_btn = ctk.CTkButton(
            title_frame,
            text="🔄 刷新",
            width=80,
            height=30,
            command=self.update_system_info
        )
        refresh_btn.pack(side="right", padx=10, pady=10)

        # 记忆系统统计
        memory_frame = ctk.CTkFrame(info_card)
        memory_frame.pack(fill="x", padx=10, pady=(0, 5))

        memory_title = ctk.CTkLabel(
            memory_frame,
            text="🧠 记忆系统",
            font=self.normal_font,
            anchor="w"
        )
        memory_title.pack(fill="x", padx=10, pady=(10, 5))

        self.memory_info = ctk.CTkTextbox(
            memory_frame,
            height=120,
            font=self.small_font
        )
        self.memory_info.pack(fill="x", padx=10, pady=(0, 10))

        # 话题权重
        topics_frame = ctk.CTkFrame(info_card)
        topics_frame.pack(fill="x", padx=10, pady=(0, 5))

        topics_title = ctk.CTkLabel(
            topics_frame,
            text="🏷️ 当前话题",
            font=self.normal_font,
            anchor="w"
        )
        topics_title.pack(fill="x", padx=10, pady=(10, 5))

        self.topics_info = ctk.CTkTextbox(
            topics_frame,
            height=100,
            font=self.small_font
        )
        self.topics_info.pack(fill="x", padx=10, pady=(0, 10))

        # 实体图谱
        entities_frame = ctk.CTkFrame(info_card)
        entities_frame.pack(fill="x", padx=10, pady=(0, 5))

        entities_title = ctk.CTkLabel(
            entities_frame,
            text="🕸️ 实体图谱",
            font=self.normal_font,
            anchor="w"
        )
        entities_title.pack(fill="x", padx=10, pady=(10, 5))

        self.entities_info = ctk.CTkTextbox(
            entities_frame,
            height=120,
            font=self.small_font
        )
        self.entities_info.pack(fill="x", padx=10, pady=(0, 10))

        # 性能指标
        performance_frame = ctk.CTkFrame(info_card)
        performance_frame.pack(fill="x", padx=10, pady=(0, 10))

        performance_title = ctk.CTkLabel(
            performance_frame,
            text="⚡ 性能指标",
            font=self.normal_font,
            anchor="w"
        )
        performance_title.pack(fill="x", padx=10, pady=(10, 5))

        self.performance_info = ctk.CTkTextbox(
            performance_frame,
            height=100,
            font=self.small_font
        )
        self.performance_info.pack(fill="x", padx=10, pady=(0, 10))

    def create_input_card(self):
        """创建输入卡片"""
        input_card = ctk.CTkFrame(self.content_frame)
        input_card.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # 输入区域标题
        input_title = ctk.CTkLabel(
            input_card,
            text="📝 输入问题",
            font=self.normal_font,
            anchor="w"
        )
        input_title.pack(fill="x", padx=10, pady=(10, 5))

        # 输入框架
        input_frame = ctk.CTkFrame(input_card)
        input_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 文本输入框
        self.question_input = ctk.CTkTextbox(
            input_frame,
            height=80,
            font=self.normal_font
        )

        # 添加提示文本
        self.question_input.insert("1.0", "请输入您的问题...")
        self.question_input.bind("<FocusIn>", self.clear_placeholder)
        self.question_input.pack(fill="both", expand=True, padx=10, pady=10)

        # 按钮区域
        button_frame = ctk.CTkFrame(input_frame)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))

        # 发送按钮
        self.send_button = ctk.CTkButton(
            button_frame,
            text="🚀 发送问题",
            width=120,
            height=35,
            command=self.send_question
        )
        self.send_button.pack(side="left", padx=5)

        # 清空输入按钮
        clear_input_btn = ctk.CTkButton(
            button_frame,
            text="🗑️ 清空输入",
            width=100,
            height=35,
            command=self.clear_input
        )
        clear_input_btn.pack(side="left", padx=5)

        # 右侧信息
        info_frame = ctk.CTkFrame(button_frame)
        info_frame.pack(side="right", padx=5)

        self.word_count_label = ctk.CTkLabel(
            info_frame,
            text="字数: 0",
            font=self.small_font
        )
        self.word_count_label.pack(side="right", padx=10)

        # 绑定输入事件
        self.question_input.bind("<KeyRelease>", self.update_word_count)
        self.question_input.bind("<Control-Return>", lambda e: self.send_question())

    def create_status_bar(self):
        """创建状态栏"""
        status_frame = ctk.CTkFrame(self.main_frame, height=30)
        status_frame.pack(fill="x", pady=(10, 0))
        status_frame.pack_propagate(False)

        # 左侧状态
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="🟢 系统就绪",
            font=self.small_font
        )
        self.status_label.pack(side="left", padx=10, pady=5)

        # 中间信息
        info_label = ctk.CTkLabel(
            status_frame,
            text="智能文档问答系统 v1.0",
            font=self.small_font,
            text_color="#1e6ba8"  # 使用主题蓝色
        )
        info_label.pack(side="left", padx=20, pady=5)

        # 右侧时间
        self.time_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=self.small_font,
            text_color="#1e6ba8"  # 使用主题蓝色
        )
        self.time_label.pack(side="right", padx=10, pady=5)

        # 更新时间显示
        self.update_time()

    def initialize_qa_system(self):
        """初始化问答系统（在后台线程中）"""
        def init_qa():
            try:
                self.update_status("🟡 正在初始化问答系统...")

                # 确保示例文档文件存在
                if not os.path.exists("local_documents.txt"):
                    create_sample_document_file()

                # 初始化问答系统
                self.qa_system = IntelligentDocumentQA()
                self.qa_system.load_documents("local_documents.txt")

                # 更新文档列表
                self.root.after(100, self.update_document_list)

                # 更新系统信息
                self.root.after(100, self.update_system_info)

                self.update_status("🟢 问答系统初始化完成")

            except Exception as e:
                error_msg = f"初始化问答系统失败: {str(e)}"
                self.update_status(f"🔴 {error_msg}")
                self.show_error(error_msg)

        # 在后台线程中初始化
        thread = threading.Thread(target=init_qa, daemon=True)
        thread.start()

    def add_welcome_message(self):
        """添加欢迎消息"""
        # 完全移除欢迎消息，保持对话区域空白
        pass

    def update_document_list(self):
        """更新文档列表显示"""
        # 清空现有文档列表
        for widget in self.doc_list_frame.winfo_children():
            widget.destroy()

        if self.qa_system and self.qa_system.documents:
            for i, doc in enumerate(self.qa_system.documents):
                doc_frame = ctk.CTkFrame(self.doc_list_frame)
                doc_frame.pack(fill="x", padx=5, pady=2)

                # 文档信息
                source = doc.metadata.get('source', f'文档{i+1}')
                paragraph = doc.metadata.get('paragraph', '')

                doc_name = f"{os.path.basename(source)}" if os.path.exists(source) else source
                if paragraph:
                    doc_name += f" (段落 {paragraph})"

                doc_label = ctk.CTkLabel(
                    doc_frame,
                    text=f"📄 {doc_name}",
                    font=self.small_font,
                    anchor="w"
                )
                doc_label.pack(side="left", padx=10, pady=5)

                # 预览按钮
                preview_btn = ctk.CTkButton(
                    doc_frame,
                    text="👁️",
                    width=30,
                    height=20,
                    command=lambda d=doc: self.preview_document(d)
                )
                preview_btn.pack(side="right", padx=5, pady=5)
        else:
            no_doc_label = ctk.CTkLabel(
                self.doc_list_frame,
                text="📂 暂无文档\n请上传文档或使用示例文档",
                font=self.small_font,
                text_color="gray"
            )
            no_doc_label.pack(pady=20)

    def preview_document(self, doc):
        """预览文档内容"""
        self.doc_preview.delete("1.0", "end")

        # 显示文档信息
        source = doc.metadata.get('source', '未知来源')
        paragraph = doc.metadata.get('paragraph', '')

        header = f"来源: {os.path.basename(source) if os.path.exists(source) else source}"
        if paragraph:
            header += f"\n段落: {paragraph}"
        header += f"\n长度: {len(doc.page_content)} 字符\n"
        header += "-" * 50 + "\n"

        self.doc_preview.insert("1.0", header)
        self.doc_preview.insert("end", doc.page_content)

    def send_question(self):
        """发送问题"""
        if self.is_processing or not self.qa_system:
            return

        question = self.question_input.get("1.0", "end-1c").strip()
        if not question or question == "请输入您的问题...":
            return

        self.is_processing = True
        self.send_button.configure(text="⏳ 处理中...", state="disabled")
        self.update_status("🟡 正在处理您的问题...")

        # 添加用户消息到对话
        self.add_user_message(question)

        # 清空输入框
        self.clear_input()

        # 在后台线程中处理问题
        def process_question():
            try:
                # 调用问答系统
                answer = self.qa_system.ask_question(question)

                # 在主线程中更新UI
                self.root.after(100, lambda: self.add_ai_message(answer))
                self.root.after(100, lambda: self.update_status("🟢 回答生成完成"))

            except Exception as e:
                error_msg = f"处理问题时出错: {str(e)}"
                self.root.after(100, lambda: self.show_error(error_msg))
                self.root.after(100, lambda: self.update_status(f"🔴 {error_msg}"))

            finally:
                self.root.after(100, self.reset_send_button)
                self.root.after(200, self.update_system_info)

        thread = threading.Thread(target=process_question, daemon=True)
        thread.start()

    def reset_send_button(self):
        """重置发送按钮状态"""
        self.is_processing = False
        self.send_button.configure(text="🚀 发送问题", state="normal")

    def add_user_message(self, message):
        """添加用户消息到对话区域"""
        message_frame = ctk.CTkFrame(self.conversation_frame)
        message_frame.pack(fill="x", padx=10, pady=5)

        # 用户头像和标识（移到右边）
        user_frame = ctk.CTkFrame(message_frame)
        user_frame.pack(fill="x", padx=10, pady=5)

        timestamp = datetime.now().strftime("%H:%M:%S")
        time_label = ctk.CTkLabel(
            user_frame,
            text=timestamp,
            font=self.small_font,
            text_color="gray"
        )
        time_label.pack(side="left", padx=5)

        user_label = ctk.CTkLabel(
            user_frame,
            text="👤 您",
            font=self.normal_font,
            text_color="#1e6ba8"
        )
        user_label.pack(side="right", padx=5)

        # 消息内容（移到右边，缩小左边距）
        msg_content = ctk.CTkTextbox(
            message_frame,
            height=60,  # 设置固定高度
            font=self.normal_font
        )
        msg_content.pack(fill="x", padx=(10, 50), pady=(0, 5))  # 减小左边距，增大右边距
        msg_content.insert("1.0", message)
        msg_content.configure(state="disabled")

        # 滚动到底部
        self.conversation_frame._parent_canvas.yview_moveto(1.0)

    def add_ai_message(self, message):
        """添加AI回复消息到对话区域"""
        message_frame = ctk.CTkFrame(self.conversation_frame)
        message_frame.pack(fill="x", padx=10, pady=5)

        # AI头像和标识（保持在左边）
        ai_frame = ctk.CTkFrame(message_frame)
        ai_frame.pack(fill="x", padx=10, pady=5)

        ai_label = ctk.CTkLabel(
            ai_frame,
            text="🤖 AI助手",
            font=self.normal_font,
            text_color="#52b788"
        )
        ai_label.pack(side="left", padx=5)

        timestamp = datetime.now().strftime("%H:%M:%S")
        time_label = ctk.CTkLabel(
            ai_frame,
            text=timestamp,
            font=self.small_font,
            text_color="gray"
        )
        time_label.pack(side="right", padx=5)

        # 消息内容（根据内容长度动态调整高度）
        # 估算需要的行数
        lines_needed = max(3, min(12, len(message) // 50 + 1))
        estimated_height = lines_needed * 20  # 每行约20像素

        msg_content = ctk.CTkTextbox(
            message_frame,
            height=estimated_height,
            font=self.normal_font
        )
        msg_content.pack(fill="both", expand=True, padx=(50, 10), pady=(0, 5))
        msg_content.insert("1.0", message)
        msg_content.configure(state="disabled")

        # 滚动到底部
        self.conversation_frame._parent_canvas.yview_moveto(1.0)

    def update_system_info(self):
        """更新系统信息显示"""
        if not self.qa_system:
            return

        try:
            # 更新记忆系统信息
            memory_info = f"记忆节点数量: {len(self.qa_system.memory_manager.memory_nodes)}\n"
            memory_info += f"最大节点数: {self.qa_system.memory_manager.max_memory_nodes}\n"
            memory_info += f"重要性阈值: {self.qa_system.memory_manager.importance_threshold}\n"
            memory_info += f"实体图谱大小: {len(self.qa_system.memory_manager.entity_graph)}"

            self.memory_info.delete("1.0", "end")
            self.memory_info.insert("1.0", memory_info)

            # 更新话题权重
            topics = self.qa_system.memory_manager.get_topic_weights()
            if topics:
                topics_text = "当前活跃话题:\n"
                for topic, weight in sorted(topics.items(), key=lambda x: x[1], reverse=True):
                    topics_text += f"• {topic}: {weight:.3f}\n"
            else:
                topics_text = "暂无活跃话题"

            self.topics_info.delete("1.0", "end")
            self.topics_info.insert("1.0", topics_text)

            # 更新实体图谱信息
            entity_graph = self.qa_system.memory_manager.entity_graph
            if entity_graph:
                entities_text = f"实体总数: {len(entity_graph)}\n\n"
                entities_text += "部分实体关系:\n"
                for i, (entity, topics) in enumerate(list(entity_graph.items())[:10]):
                    topics_str = ", ".join(list(topics)[:3])
                    entities_text += f"• {entity}: {topics_str}\n"
            else:
                entities_text = "暂无实体关系"

            self.entities_info.delete("1.0", "end")
            self.entities_info.insert("1.0", entities_text)

            # 更新性能指标
            perf_text = f"文档数量: {len(self.qa_system.documents)}\n"
            perf_text += f"文档分块: {len(self.qa_system.document_chunks)}\n"
            perf_text += f"对话轮次: {len(self.qa_system.message_history.messages) // 2}\n"
            perf_text += f"话题历史: {len(self.qa_system.memory_manager.conversation_topics)}"

            self.performance_info.delete("1.0", "end")
            self.performance_info.insert("1.0", perf_text)

        except Exception as e:
            error_msg = f"更新系统信息时出错: {str(e)}"
            print(error_msg)

    def update_word_count(self, event=None):
        """更新字数统计"""
        text = self.question_input.get("1.0", "end-1c")
        word_count = len(text)
        self.word_count_label.configure(text=f"字数: {word_count}")

    def update_status(self, message):
        """更新状态栏"""
        self.status_label.configure(text=message)

    def update_time(self):
        """更新时间显示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.root.after(1000, self.update_time)  # 每秒更新一次

    def clear_placeholder(self, event=None):
        """清空占位符文本"""
        current_text = self.question_input.get("1.0", "1.0").strip()
        if current_text == "请输入您的问题...":
            self.question_input.delete("1.0", "end")
            self.question_input.unbind("<FocusIn>", self.clear_placeholder)

    def clear_input(self):
        """清空输入框"""
        self.question_input.delete("1.0", "end")
        # 重新插入占位符
        self.question_input.insert("1.0", "请输入您的问题...")
        self.question_input.bind("<FocusIn>", self.clear_placeholder)
        self.update_word_count()

    def clear_conversation(self):
        """清空对话历史"""
        result = messagebox.askyesno("确认", "确定要清空所有对话历史吗？")
        if result:
            # 清空对话区域
            for widget in self.conversation_frame.winfo_children():
                widget.destroy()

            # 重新添加简洁的欢迎消息
            self.add_welcome_message()

            # 清空问答系统的对话历史
            if self.qa_system:
                self.qa_system.message_history.clear()

            self.update_status("对话历史已清空")  # 移除灰色图标

    def export_conversation(self):
        """导出对话历史"""
        if not self.qa_system:
            messagebox.showwarning("提示", "请先初始化问答系统")
            return

        from tkinter import filedialog, messagebox

        file_path = filedialog.asksaveasfilename(
            title="导出对话历史",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("智能文档问答系统 - 对话历史\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    # 写入对话历史
                    messages = self.qa_system.message_history.messages
                    for i, message in enumerate(messages, 1):
                        from langchain_core.messages import HumanMessage, AIMessage
                        if isinstance(message, HumanMessage):
                            f.write(f"[用户] {message.content}\n\n")
                        elif isinstance(message, AIMessage):
                            f.write(f"[AI助手] {message.content}\n\n")
                        else:
                            f.write(f"[消息{i}] {message.content}\n\n")

                messagebox.showinfo("成功", f"对话历史已导出到: {file_path}")
                self.update_status("🟢 对话历史导出成功")

            except Exception as e:
                error_msg = f"导出对话历史时出错: {str(e)}"
                messagebox.showerror("错误", error_msg)
                self.update_status(f"🔴 {error_msg}")

    def upload_document(self):
        """上传文档"""
        file_path = filedialog.askopenfilename(
            title="选择文档文件",
            filetypes=[
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                self.update_status(f"🟡 正在加载文档: {os.path.basename(file_path)}")

                # 加载文档
                self.qa_system.load_documents(file_path)

                # 更新文档列表
                self.update_document_list()

                # 更新系统信息
                self.update_system_info()

                self.update_status("🟢 文档加载完成")
                messagebox.showinfo("成功", f"文档已成功加载: {os.path.basename(file_path)}")

            except Exception as e:
                error_msg = f"加载文档时出错: {str(e)}"
                messagebox.showerror("错误", error_msg)
                self.update_status(f"🔴 {error_msg}")

    def toggle_theme(self):
        """切换主题"""
        if self.current_theme == "light":
            ctk.set_appearance_mode("dark")
            self.current_theme = "dark"
            self.theme_button.configure(text="☀️ 亮色主题")
        else:
            ctk.set_appearance_mode("light")
            self.current_theme = "light"
            self.theme_button.configure(text="🌙 暗色主题")

    def open_settings(self):
        """打开设置窗口"""
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("系统设置")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()

        # 设置标题
        title_label = ctk.CTkLabel(
            settings_window,
            text="⚙️ 系统设置",
            font=self.heading_font,
            text_color="#1e6ba8"
        )
        title_label.pack(pady=20)

        # 模型设置
        model_frame = ctk.CTkFrame(settings_window)
        model_frame.pack(fill="x", padx=20, pady=10)

        model_label = ctk.CTkLabel(
            model_frame,
            text="🤖 模型设置",
            font=self.normal_font
        )
        model_label.pack(anchor="w", padx=10, pady=(10, 5))

        # 温度设置
        temp_frame = ctk.CTkFrame(model_frame)
        temp_frame.pack(fill="x", padx=10, pady=5)

        temp_label = ctk.CTkLabel(temp_frame, text="温度:", width=60)
        temp_label.pack(side="left", padx=5)

        temp_var = tk.DoubleVar(value=0.7)
        temp_slider = ctk.CTkSlider(
            temp_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=10,
            variable=temp_var
        )
        temp_slider.pack(side="left", fill="x", expand=True, padx=5)

        temp_value = ctk.CTkLabel(temp_frame, text="0.7", width=40)
        temp_value.pack(side="left", padx=5)

        def update_temp_label(value):
            temp_value.configure(text=f"{float(value):.1f}")

        temp_slider.configure(command=update_temp_label)

        # 记忆设置
        memory_frame = ctk.CTkFrame(settings_window)
        memory_frame.pack(fill="x", padx=20, pady=10)

        memory_label = ctk.CTkLabel(
            memory_frame,
            text="🧠 记忆设置",
            font=self.normal_font
        )
        memory_label.pack(anchor="w", padx=10, pady=(10, 5))

        # 最大记忆节点
        nodes_frame = ctk.CTkFrame(memory_frame)
        nodes_frame.pack(fill="x", padx=10, pady=5)

        nodes_label = ctk.CTkLabel(nodes_frame, text="最大节点数:", width=80)
        nodes_label.pack(side="left", padx=5)

        nodes_var = tk.IntVar(value=50)
        nodes_entry = ctk.CTkEntry(nodes_frame, textvariable=nodes_var, width=80)
        nodes_entry.pack(side="left", padx=5)

        # 重要性阈值
        threshold_frame = ctk.CTkFrame(memory_frame)
        threshold_frame.pack(fill="x", padx=10, pady=5)

        threshold_label = ctk.CTkLabel(threshold_frame, text="重要性阈值:", width=80)
        threshold_label.pack(side="left", padx=5)

        threshold_var = tk.DoubleVar(value=0.3)
        threshold_entry = ctk.CTkEntry(threshold_frame, textvariable=threshold_var, width=80)
        threshold_entry.pack(side="left", padx=5)

        # 文档设置
        doc_frame = ctk.CTkFrame(settings_window)
        doc_frame.pack(fill="x", padx=20, pady=10)

        doc_label = ctk.CTkLabel(
            doc_frame,
            text="📁 文档设置",
            font=self.normal_font
        )
        doc_label.pack(anchor="w", padx=10, pady=(10, 5))

        # 检索数量
        retrieve_frame = ctk.CTkFrame(doc_frame)
        retrieve_frame.pack(fill="x", padx=10, pady=5)

        retrieve_label = ctk.CTkLabel(retrieve_frame, text="检索数量:", width=80)
        retrieve_label.pack(side="left", padx=5)

        retrieve_var = tk.IntVar(value=3)
        retrieve_entry = ctk.CTkEntry(retrieve_frame, textvariable=retrieve_var, width=80)
        retrieve_entry.pack(side="left", padx=5)

        # 按钮区域
        button_frame = ctk.CTkFrame(settings_window)
        button_frame.pack(fill="x", padx=20, pady=20)

        save_btn = ctk.CTkButton(
            button_frame,
            text="💾 保存设置",
            command=lambda: self.save_settings(settings_window)
        )
        save_btn.pack(side="left", padx=5)

        reset_btn = ctk.CTkButton(
            button_frame,
            text="🔄 恢复默认",
            command=lambda: self.reset_settings(settings_window)
        )
        reset_btn.pack(side="left", padx=5)

        cancel_btn = ctk.CTkButton(
            button_frame,
            text="❌ 取消",
            command=settings_window.destroy
        )
        cancel_btn.pack(side="right", padx=5)

    def save_settings(self, window):
        """保存设置"""
        # 这里可以添加设置保存逻辑
        messagebox.showinfo("成功", "设置已保存")
        window.destroy()

    def reset_settings(self, window):
        """恢复默认设置"""
        # 这里可以添加设置重置逻辑
        messagebox.showinfo("成功", "设置已恢复默认")
        window.destroy()

    def show_help(self):
        """显示帮助信息"""
        help_window = ctk.CTkToplevel(self.root)
        help_window.title("帮助文档")
        help_window.geometry("600x500")
        help_window.transient(self.root)

        # 帮助内容
        help_text = """
🤖 智能文档问答系统 - 使用帮助

📋 功能介绍：
• 智能记忆管理：系统能够记住对话历史，理解上下文关系
• 动态文档检索：基于TF-IDF算法智能匹配相关文档内容
• 多轮对话理解：解析代词引用和省略信息
• 实时性能监控：追踪记忆系统、话题权重和实体图谱

🔧 使用方法：
1. 文档管理：点击"上传文档"按钮添加你的文档文件
2. 开始对话：在底部输入框中输入问题，点击"发送问题"
3. 查看信息：右侧面板实时显示系统状态和统计信息
4. 导出对话：点击"导出"按钮保存对话历史

⌨️ 快捷键：
• Ctrl+Enter：发送问题
• Esc：清空输入框

💡 使用技巧：
• 系统支持多轮对话，可以引用之前讨论的内容
• 文档会在后台自动分块和索引
• 可以通过右侧监控面板了解系统工作状态
• 支持明/暗主题切换

🔧 配置要求：
• Python 3.8+
• CustomTkinter库
• 有效的API密钥配置

❓ 常见问题：
Q: 如何添加自己的文档？
A: 点击文档管理区域的"上传文档"按钮，选择.txt文件即可。

Q: 系统支持哪些文档格式？
A: 目前主要支持.txt格式，后续将支持更多格式。

Q: 对话历史会被保存吗？
A: 对话历史会保存在内存中，重启程序后会清空，建议使用导出功能保存。

Q: 如何提高回答准确性？
A: 提供清晰、具体的问题，确保文档内容相关且完整。

如需更多帮助，请联系技术支持。
        """

        # 创建滚动文本框
        help_scroll = ctk.CTkScrollableFrame(help_window)
        help_scroll.pack(fill="both", expand=True, padx=20, pady=20)

        help_label = ctk.CTkLabel(
            help_scroll,
            text=help_text,
            font=self.small_font,
            justify="left",
            anchor="w"
        )
        help_label.pack(fill="both", expand=True)

        # 关闭按钮
        close_btn = ctk.CTkButton(
            help_window,
            text="关闭",
            command=help_window.destroy,
            width=100
        )
        close_btn.pack(pady=10)

    def show_error(self, message):
        """显示错误消息"""
        messagebox.showerror("错误", message)

    def on_closing(self):
        """窗口关闭事件"""
        if messagebox.askokcancel("退出", "确定要退出智能文档问答系统吗？"):
            self.root.destroy()

    def run(self):
        """运行应用程序"""
        self.root.mainloop()

def main():
    """主函数"""
    app = DocumentQAGUI()
    app.run()

if __name__ == "__main__":
    main()