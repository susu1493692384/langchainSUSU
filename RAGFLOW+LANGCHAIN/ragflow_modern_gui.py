#!/usr/bin/env python3
"""
RAGFlow + LangChain 现代化GUI前端
提供现代化的用户界面和交互体验
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
from PIL import Image, ImageTk
import re

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("警告: 未安装python-dotenv，将使用默认配置")

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ragflow_langchain_integration import RAGFlowLangChainApp, RAGFlowAPIConnector
except ImportError:
    print("错误: 无法导入RAGFlow集成模块")
    print("请确保ragflow_langchain_integration.py在同一目录下")
    sys.exit(1)

class ModernStyle:
    """现代化样式配置"""

    @staticmethod
    def setup_theme():
        """设置现代化主题"""
        style = ttk.Style()

        # 尝试使用现代主题
        try:
            style.theme_use('clam')
        except:
            try:
                style.theme_use('alt')
            except:
                style.theme_use('default')

        # 配置颜色方案
        colors = {
            'bg': '#f8f9fa',           # 背景色
            'fg': '#212529',           # 前景色
            'select_bg': '#007bff',    # 选中背景色
            'select_fg': 'white',      # 选中前景色
            'button_bg': '#007bff',    # 按钮背景色
            'button_fg': 'white',      # 按钮前景色
            'accent': '#17a2b8',       # 强调色
            'success': '#28a745',      # 成功色
            'warning': '#ffc107',      # 警告色
            'danger': '#dc3545',       # 危险色
            'border': '#dee2e6',       # 边框色
            'shadow': '#6c757d',       # 阴影色
        }

        # 配置各种组件样式
        style.configure('TFrame', background=colors['bg'])
        style.configure('TLabel', background=colors['bg'], foreground=colors['fg'], font=('黑体', 9))
        style.configure('TButton',
                       font=('黑体', 9, 'bold'),
                       padding=(12, 6),
                       relief=tk.FLAT,
                       borderwidth=1)
        style.map('TButton',
                 background=[('active', colors['select_bg']),
                           ('pressed', colors['select_bg'])],
                 foreground=[('active', colors['select_fg']),
                           ('pressed', colors['select_fg'])])

        style.configure('Primary.TButton',
                       background=colors['button_bg'],
                       foreground=colors['button_fg'])
        style.map('Primary.TButton',
                 background=[('active', '#0056b3'),
                           ('pressed', '#004085')])

        style.configure('Success.TButton',
                       background=colors['success'],
                       foreground='white')
        style.map('Success.TButton',
                 background=[('active', '#218838'),
                           ('pressed', '#1e7e34')])

        style.configure('Danger.TButton',
                       background=colors['danger'],
                       foreground='white')
        style.map('Danger.TButton',
                 background=[('active', '#c82333'),
                           ('pressed', '#bd2130')])

        style.configure('TEntry',
                       fieldbackground='white',
                       borderwidth=1,
                       relief=tk.SOLID,
                       font=('黑体', 9))

        style.configure('TLabelframe',
                       background=colors['bg'],
                       borderwidth=1,
                       relief=tk.SOLID)
        style.configure('TLabelframe.Label',
                       background=colors['bg'],
                       foreground=colors['fg'],
                       font=('黑体', 10, 'bold'))

        style.configure('TCombobox',
                       fieldbackground='white',
                       borderwidth=1,
                       relief=tk.SOLID,
                       font=('黑体', 9))

        style.configure('TNotebook',
                       background=colors['bg'],
                       borderwidth=0)
        style.configure('TNotebook.Tab',
                       background=colors['border'],
                       foreground=colors['fg'],
                       padding=(20, 8),
                       font=('黑体', 9))
        style.map('TNotebook.Tab',
                 background=[('selected', colors['bg']),
                           ('active', '#e9ecef')])

        return colors

class ModernChatMessage:
    """现代化聊天消息类"""
    def __init__(self, message_type: str, content: str, timestamp: str = None):
        self.message_type = message_type  # "user" 或 "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M:%S")
        self.widget = None  # 存储消息的UI组件

class ModernChatWidget(tk.Frame):
    """现代化聊天显示组件"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # 配置现代化样式
        ModernStyle.setup_theme()
        self.colors = ModernStyle.setup_theme()

        # 设置背景色
        self.config(bg=self.colors['bg'])

        # 创建滚动区域
        self.canvas = tk.Canvas(self, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.colors['bg'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # 布局
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 消息容器 - 使用更现代的背景色
        self.message_container = tk.Frame(self.scrollable_frame, bg='#FFFFFF')
        self.message_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 消息历史
        self.messages: List[ModernChatMessage] = []

        # 添加欢迎消息
        self.add_system_message("🚀 欢迎使用 RAGFlow 智能问答系统",
                               "开始使用前，请先连接到 RAGFlow 服务并选择知识库。祝您使用愉快！")

    def add_message(self, message_type: str, content: str, title: str = None):
        """添加消息到聊天历史"""
        message = ModernChatMessage(message_type, content)
        self.messages.append(message)

        # 创建消息UI
        message_frame = self._create_message_widget(message, title)
        message.widget = message_frame

        # 添加到容器
        message_frame.pack(fill=tk.X, pady=12, padx=8)

        # 自动滚动到底部
        self.after(100, self._scroll_to_bottom)

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.add_message("user", content, "您")

    def add_assistant_message(self, content: str):
        """添加助手回复"""
        self.add_message("assistant", content, "AI助手")

    def add_system_message(self, content: str, subtitle: str = None):
        """添加系统消息"""
        self.add_message("system", content, subtitle or "系统")

    def add_error_message(self, content: str):
        """添加错误消息"""
        self.add_message("error", content, "❌ 错误")

    def _create_message_widget(self, message: ModernChatMessage, title: str = None) -> tk.Frame:
        """创建现代化的消息UI组件"""
        # 消息主框架
        msg_frame = tk.Frame(self.message_container, bg='#FFFFFF')

        # 根据消息类型选择样式
        if message.message_type == "user":
            # 用户消息 - 现代蓝色，右对齐
            content_frame = self._create_bubble_frame(
                msg_frame,
                title or "👤 您",
                message.timestamp,
                bg_color="#5856D6",  # 现代紫色蓝色
                text_color="white",
                align="right",
                content=message.content
            )
            content_frame.pack(side=tk.RIGHT, anchor=tk.E, fill=tk.X, padx=(0, 60))

        elif message.message_type == "assistant":
            # AI助手回复 - 现代绿色，左对齐
            content_frame = self._create_bubble_frame(
                msg_frame,
                title or "🤖 AI助手",
                message.timestamp,
                bg_color="#34C759",  # 现代绿色
                text_color="white",
                align="left",
                content=message.content
            )
            content_frame.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, padx=(60, 0))

        elif message.message_type == "error":
            # 错误消息 - 现代红色，居中
            content_frame = self._create_bubble_frame(
                msg_frame,
                title or "⚠️ 错误",
                message.timestamp,
                bg_color="#FF3B30",  # 现代红色
                text_color="white",
                align="center",
                content=message.content
            )
            content_frame.pack(side=tk.TOP, anchor=tk.CENTER, fill=tk.X, padx=120)

        else:  # system
            # 系统消息 - 现代化样式
            content_frame = self._create_bubble_frame(
                msg_frame,
                title or "ℹ️ 系统",
                message.timestamp,
                bg_color="#F2F2F7",  # 现代浅灰色
                text_color="#8E8E93",  # 现代灰色文字
                align="center",
                content=message.content
            )
            content_frame.pack(side=tk.TOP, anchor=tk.CENTER, fill=tk.X, padx=80)

        return msg_frame

    def _create_bubble_frame(self, parent, title: str, timestamp: str, bg_color: str,
                           text_color: str, align: str = "left", content: str = "") -> tk.Frame:
        """创建现代化的聊天气泡框架"""
        # 外层容器 - 创建阴影效果
        shadow_container = tk.Frame(parent, bg='#F0F0F0')  # 浅灰色阴影

        # 主气泡容器
        bubble = tk.Frame(
            shadow_container,
            bg=bg_color,
            relief=tk.SOLID,
            borderwidth=1,
            highlightthickness=0
        )

        # 设置阴影效果
        shadow_container.configure(borderwidth=1, relief=tk.FLAT)
        bubble.configure(relief=tk.RAISED, borderwidth=1)

        # 主容器布局
        bubble.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # 内容容器
        content_container = tk.Frame(bubble, bg=bg_color)
        content_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # 标题和时间戳容器
        header_frame = tk.Frame(content_container, bg=bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 6))

        # 标题
        title_label = tk.Label(
            header_frame,
            text=title,
            bg=bg_color,
            fg=text_color,
            font=('黑体', 9, 'bold'),
            anchor=tk.W
        )
        title_label.pack(side=tk.LEFT)

        # 时间戳
        time_label = tk.Label(
            header_frame,
            text=timestamp,
            bg=bg_color,
            fg=self._adjust_color_brightness(text_color, 0.7),
            font=('黑体', 7),
            anchor=tk.E
        )
        time_label.pack(side=tk.RIGHT)

        # 微妙的分隔线
        if content.strip():  # 只有有内容时才显示分隔线
            separator_frame = tk.Frame(content_container, bg=self._lighten_color(bg_color, 0.2), height=1)
            separator_frame.pack(fill=tk.X, pady=(6, 6))

        # 内容区域 - 使用黑体显示
        content_text = tk.Text(
            content_container,
            bg=bg_color,
            fg=text_color,
            font=('黑体', 10),
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            height=1,
            width=45,
            spacing1=0,
            spacing2=0,
            spacing3=0,
            selectbackground=self._darken_color(bg_color, 0.2),
            selectforeground=text_color
        )
        content_text.pack(fill=tk.BOTH, expand=True)

        # 插入消息内容
        if content.strip():
            content_text.insert(tk.END, content)

        # 设置为只读
        content_text.configure(state=tk.DISABLED)

        # 调整文本高度
        if content.strip():
            bubble.after(10, lambda: self._adjust_text_height(content_text))

        return shadow_container

    def _lighten_color(self, color: str, factor: float) -> str:
        """使颜色变亮"""
        if color.startswith('#'):
            # 十六进制颜色
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            new_rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
            return '#%02x%02x%02x' % new_rgb
        return color

    def _darken_color(self, color: str, factor: float) -> str:
        """使颜色变暗"""
        if color.startswith('#'):
            # 十六进制颜色
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            new_rgb = tuple(max(0, int(c * (1 - factor))) for c in rgb)
            return '#%02x%02x%02x' % new_rgb
        return color

    def _adjust_color_brightness(self, color: str, factor: float) -> str:
        """调整颜色亮度"""
        if color.startswith('#'):
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            new_rgb = tuple(max(0, min(255, int(c * factor))) for c in rgb)
            return '#%02x%02x%02x' % new_rgb
        return color

    def _adjust_text_height(self, text_widget):
        """调整文本控件高度以适应内容"""
        try:
            text_widget.config(state=tk.NORMAL)
            content = text_widget.get(1.0, tk.END).strip()

            # 计算需要的行数
            line_count = len(content.split('\n'))
            char_count = len(content)

            # 根据字符数和行数估算高度
            if line_count == 1 and char_count < 30:
                height = 1
            elif line_count == 1:
                height = 2
            elif line_count < 5:
                height = line_count + 1
            else:
                height = 6  # 最大高度

            text_widget.config(height=height, state=tk.DISABLED)

        except:
            text_widget.config(state=tk.DISABLED)

    def clear_history(self):
        """清空聊天历史"""
        # 清空UI
        for widget in self.message_container.winfo_children():
            widget.destroy()

        # 清空数据
        self.messages.clear()

        # 添加欢迎消息
        self.add_system_message("🆕 对话历史已清空",
                               "您可以开始新的对话了。期待为您提供帮助！")

    def _scroll_to_bottom(self):
        """自动滚动到底部"""
        try:
            self.canvas.update_idletasks()
            self.canvas.yview_moveto(1.0)
        except:
            pass

class ModernInputWidget(tk.Frame):
    """现代化输入组件"""

    def __init__(self, parent, on_send_callback=None, **kwargs):
        super().__init__(parent, **kwargs)

        # 配置样式
        self.colors = ModernStyle.setup_theme()
        self.config(bg=self.colors['bg'])
        self.on_send_callback = on_send_callback

        # 创建现代化输入区域
        self._create_input_area()

        # 初始状态
        self.set_enabled(False)

    def _create_input_area(self):
        """创建现代化输入区域"""
        # 主容器
        main_frame = tk.Frame(self, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 输入框容器
        input_container = tk.Frame(main_frame, bg=self.colors['bg'])
        input_container.pack(fill=tk.X, pady=(0, 10))

        # 输入框样式设置
        self.text_input = tk.Text(
            input_container,
            wrap=tk.WORD,
            height=4,
            font=('黑体', 10),
            relief=tk.SOLID,
            borderwidth=1,
            bg='white',
            fg=self.colors['fg'],
            selectbackground=self.colors['select_bg'],
            selectforeground=self.colors['select_fg'],
            padx=12,
            pady=8,
            insertbackground=self.colors['fg'],
            insertwidth=1
        )
        self.text_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # 绑定快捷键和事件
        self.text_input.bind("<Control-Return>", self._on_enter_send)
        self.text_input.bind("<Shift-Return>", lambda e: None)  # 允许换行
        self.text_input.bind("<KeyRelease>", self._on_key_release)

        # 添加占位符
        self._add_placeholder()

        # 按钮容器
        button_container = tk.Frame(input_container, bg=self.colors['bg'])
        button_container.pack(side=tk.RIGHT, fill=tk.Y)

        # 清空按钮
        self.clear_button = ttk.Button(
            button_container,
            text="🗑️",
            command=self.clear_input,
            width=3,
            style='Danger.TButton'
        )
        self.clear_button.pack(pady=(0, 5))

        # 发送按钮
        self.send_button = ttk.Button(
            button_container,
            text="发送\n✈️",
            command=self.send_message,
            width=8,
            style='Primary.TButton'
        )
        self.send_button.pack()

        # 状态栏
        self.status_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        self.status_frame.pack(fill=tk.X)

        self.status_label = tk.Label(
            self.status_frame,
            text="Ctrl+Enter 发送消息 | Shift+Enter 换行",
            bg=self.colors['bg'],
            fg=self.colors['shadow'],
            font=('黑体', 8)
        )
        self.status_label.pack(side=tk.LEFT)

        # 字数统计
        self.char_count_label = tk.Label(
            self.status_frame,
            text="0 / 2000 字符",
            bg=self.colors['bg'],
            fg=self.colors['shadow'],
            font=('黑体', 8)
        )
        self.char_count_label.pack(side=tk.RIGHT)

    def _add_placeholder(self):
        """添加输入框占位符"""
        placeholder_text = "请输入您的问题..."
        self.text_input.insert(tk.END, placeholder_text)
        self.text_input.config(foreground=self.colors['shadow'])

        # 绑定焦点事件
        self.text_input.bind("<FocusIn>", self._on_focus_in)
        self.text_input.bind("<FocusOut>", self._on_focus_out)

        self.has_placeholder = True

    def _on_focus_in(self, event):
        """焦点获得时处理占位符"""
        if self.has_placeholder:
            self.text_input.delete(1.0, tk.END)
            self.text_input.config(foreground=self.colors['fg'])
            self.has_placeholder = False

    def _on_focus_out(self, event):
        """焦点失去时处理占位符"""
        if not self.text_input.get(1.0, tk.END).strip() and not self.has_placeholder:
            self.text_input.insert(tk.END, "请输入您的问题...")
            self.text_input.config(foreground=self.colors['shadow'])
            self.has_placeholder = True

    def _on_key_release(self, event):
        """按键释放时更新状态"""
        content = self.text_input.get(1.0, tk.END).strip()
        char_count = len(content)

        # 更新字数统计
        self.char_count_label.config(text=f"{char_count} / 2000 字符")

        # 超过字数限制时变色
        if char_count > 2000:
            self.char_count_label.config(foreground=self.colors['danger'])
        else:
            self.char_count_label.config(foreground=self.colors['shadow'])

    def send_message(self):
        """发送消息"""
        content = self.text_input.get(1.0, tk.END).strip()

        if content and not self.has_placeholder and self.on_send_callback:
            # 检查字数限制
            if len(content) > 2000:
                messagebox.showwarning("字数限制", "消息内容不能超过2000个字符")
                return

            self.on_send_callback(content)
            self.clear_input()

    def _on_enter_send(self, event):
        """回车键发送消息"""
        if event.state & 0x4:  # Ctrl键
            self.send_message()
            return "break"
        return None

    def clear_input(self):
        """清空输入框"""
        self.text_input.delete(1.0, tk.END)
        self._on_focus_out(None)

    def set_enabled(self, enabled: bool):
        """设置输入框启用状态"""
        if enabled:
            self.text_input.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            self.status_label.config(text="Ctrl+Enter 发送消息 | Shift+Enter 换行")
        else:
            self.text_input.config(state=tk.DISABLED)
            self.send_button.config(state=tk.DISABLED)
            self.clear_button.config(state=tk.DISABLED)
            self.status_label.config(text="请先连接RAGFlow服务")

    def get_input(self) -> str:
        """获取输入内容"""
        if self.has_placeholder:
            return ""
        return self.text_input.get(1.0, tk.END).strip()

class ModernConfigWidget(tk.Frame):
    """现代化配置管理组件"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # 配置样式
        self.colors = ModernStyle.setup_theme()
        self.config(bg=self.colors['bg'])

        # 从环境变量加载默认配置
        self.config_data = {
            "ragflow_url": os.getenv("RAGFLOW_API_URL", "http://localhost:9380"),
            "ragflow_api_key": os.getenv("RAGFLOW_API_KEY", ""),
            "llm_model": os.getenv("LLM_MODEL", "glm-4.5"),
            "top_k": int(os.getenv("TOP_K", "5")),
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        }

        # 创建现代化界面
        self._create_ui()

        # 尝试从文件加载配置
        self.load_config()

    def _create_ui(self):
        """创建现代化配置界面"""
        # 主框架
        main_frame = ttk.LabelFrame(self, text="⚙️ RAGFlow配置", padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 配置字段
        fields = [
            ("🌐 RAGFlow服务地址", "ragflow_url", "RAGFlow API服务地址"),
            ("🔑 API密钥", "ragflow_api_key", "您的RAGFlow API密钥"),
            ("🤖 LLM模型", "llm_model", "选择语言模型"),
            ("📊 检索结果数量", "top_k", "返回相关文档数量"),
            ("🎯 相似度阈值", "similarity_threshold", "文档相似度最低要求")
        ]

        self.vars = {}

        for i, (label, key, tooltip) in enumerate(fields):
            # 标签
            ttk.Label(main_frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, pady=8, padx=(0, 10))

            if key == "ragflow_api_key":
                # API密钥 - 密码输入框
                var = tk.StringVar(value=self.config_data[key])
                entry = ttk.Entry(main_frame, textvariable=var, width=25, show="*")
                self.vars[key] = var
            elif key == "llm_model":
                # LLM模型 - 下拉框
                var = tk.StringVar(value=self.config_data[key])
                combo = ttk.Combobox(main_frame, textvariable=var, width=23, state="readonly")
                combo['values'] = ("glm-4.5", "gpt-3.5-turbo", "gpt-4", "claude-3-sonnet")
                self.vars[key] = var
                entry = combo
            elif key == "top_k":
                # 检索数量 - 数字选择框
                var = tk.IntVar(value=self.config_data[key])
                spinbox = ttk.Spinbox(main_frame, from_=1, to=20, textvariable=var, width=24)
                self.vars[key] = var
                entry = spinbox
            elif key == "similarity_threshold":
                # 相似度阈值 - 滑块
                var = tk.DoubleVar(value=self.config_data[key])
                self.vars[key] = var

                # 创建带滑块的框架
                slider_frame = tk.Frame(main_frame)
                slider_frame.grid(row=i, column=1, sticky=tk.EW, pady=8)

                slider = ttk.Scale(
                    slider_frame,
                    from_=0.1,
                    to=1.0,
                    variable=var,
                    orient=tk.HORIZONTAL,
                    length=200
                )
                slider.pack(side=tk.LEFT)

                value_label = ttk.Label(slider_frame, text=f"{var.get():.1f}")
                value_label.pack(side=tk.LEFT, padx=(10, 0))

                # 更新显示值
                def update_value(*args):
                    value_label.config(text=f"{var.get():.1f}")
                var.trace('w', update_value)

                entry = slider_frame
            else:
                # 普通输入框
                var = tk.StringVar(value=self.config_data[key])
                entry = ttk.Entry(main_frame, textvariable=var, width=25)
                self.vars[key] = var

            if key != "similarity_threshold":
                entry.grid(row=i, column=1, sticky=tk.EW, pady=8, padx=(0, 10))

        # 配置列权重
        main_frame.columnconfigure(1, weight=1)

        # 按钮容器
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=len(fields), column=0, columnspan=2, pady=(20, 0))

        # 现代化按钮
        ttk.Button(
            button_frame,
            text="💾 保存配置",
            command=self.save_config,
            style='Success.TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame,
            text="📂 加载配置",
            command=self.load_config
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame,
            text="🔄 重置配置",
            command=self.reset_config,
            style='Danger.TButton'
        ).pack(side=tk.LEFT)

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        config = {}
        for key, var in self.vars.items():
            config[key] = var.get()
        return config

    def save_config(self):
        """保存配置到文件"""
        try:
            self.config_data = self.get_config()
            config_file = "gui_config.json"

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("✅ 成功", f"配置已保存到 {config_file}")
        except Exception as e:
            messagebox.showerror("❌ 错误", f"保存配置失败: {e}")

    def load_config(self):
        """从文件加载配置"""
        try:
            config_file = "gui_config.json"
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # 更新配置数据
                self.config_data.update(loaded_config)

                # 更新界面（如果vars已初始化）
                if hasattr(self, 'vars'):
                    for key, var in self.vars.items():
                        if key in self.config_data:
                            var.set(self.config_data[key])

                    messagebox.showinfo("✅ 成功", f"配置已从 {config_file} 加载")
            else:
                # 只在UI已创建时显示消息
                if hasattr(self, 'vars'):
                    messagebox.showinfo("ℹ️ 提示", "配置文件不存在，使用默认配置")
        except Exception as e:
            # 只在UI已创建时显示错误消息
            if hasattr(self, 'vars'):
                messagebox.showerror("❌ 错误", f"加载配置失败: {e}")

    def reset_config(self):
        """重置配置为默认值"""
        if messagebox.askyesno("⚠️ 确认", "确定要重置配置为默认值吗？"):
            self.config_data = {
                "ragflow_url": "http://localhost:9380",
                "ragflow_api_key": "",
                "llm_model": "glm-4.5",
                "top_k": 5,
                "similarity_threshold": 0.7
            }

            # 更新界面
            for key, var in self.vars.items():
                if key in self.config_data:
                    var.set(self.config_data[key])

            messagebox.showinfo("✅ 成功", "配置已重置为默认值")

class ModernKnowledgeBaseWidget(tk.Frame):
    """现代化知识库选择组件"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # 配置样式
        self.colors = ModernStyle.setup_theme()
        self.config(bg=self.colors['bg'])

        self.knowledge_bases = []
        self.selected_kb = None

        self._create_ui()

    def _create_ui(self):
        """创建现代化知识库选择界面"""
        # 主框架
        main_frame = ttk.LabelFrame(self, text="📚 知识库", padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 工具栏
        toolbar = tk.Frame(main_frame, bg=self.colors['bg'])
        toolbar.pack(fill=tk.X, pady=(0, 10))

        # 搜索框
        tk.Label(toolbar, text="🔍", bg=self.colors['bg'], font=('黑体', 10)).pack(side=tk.LEFT, padx=(0, 5))

        self.search_var = tk.StringVar()
        self.search_var.trace('w', self._filter_knowledge_bases)

        self.search_entry = ttk.Entry(
            toolbar,
            textvariable=self.search_var,
            width=20
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # 刷新按钮
        ttk.Button(
            toolbar,
            text="🔄 刷新",
            command=self.refresh_knowledge_bases,
            width=10
        ).pack(side=tk.RIGHT)

        # 列表框容器
        list_container = tk.Frame(main_frame, bg=self.colors['bg'])
        list_container.pack(fill=tk.BOTH, expand=True)

        # 创建现代化的列表框
        self.listbox = tk.Listbox(
            list_container,
            height=12,
            font=('黑体', 9),
            bg='white',
            fg=self.colors['fg'],
            selectbackground=self.colors['select_bg'],
            selectforeground=self.colors['select_fg'],
            activestyle='none',
            relief=tk.SOLID,
            borderwidth=1,
            highlightthickness=0
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        # 状态标签
        self.status_label = tk.Label(
            main_frame,
            text="📦 共 0 个知识库",
            bg=self.colors['bg'],
            fg=self.colors['shadow'],
            font=('黑体', 8)
        )
        self.status_label.pack(pady=(10, 0))

    def _filter_knowledge_bases(self, *args):
        """过滤知识库列表"""
        search_text = self.search_var.get().lower()

        # 清空列表
        self.listbox.delete(0, tk.END)

        # 过滤并重新添加
        filtered_count = 0
        for i, kb in enumerate(self.knowledge_bases):
            if isinstance(kb, str):
                display_text = kb
                if search_text in display_text.lower():
                    self.listbox.insert(tk.END, display_text)
                    filtered_count += 1
            elif isinstance(kb, dict):
                name = kb.get('name', '未知')
                desc = kb.get('description', '')
                doc_count = kb.get('document_count', 0)

                display_text = f"📄 {name}"
                if desc:
                    display_text += f" - {desc}"
                if doc_count:
                    display_text += f" ({doc_count}个文档)"

                if search_text in name.lower() or search_text in desc.lower():
                    self.listbox.insert(tk.END, display_text)
                    filtered_count += 1

        # 更新状态
        self.status_label.config(text=f"📦 共 {filtered_count} 个知识库")

    def update_knowledge_bases(self, kbs: List[Any]):
        """更新知识库列表"""
        self.knowledge_bases = kbs
        self._filter_knowledge_bases()  # 应用当前过滤条件

        # 更新状态
        self.status_label.config(text=f"📦 共 {len(kbs)} 个知识库")

    def get_selected_knowledge_base(self) -> Optional[str]:
        """获取选中的知识库"""
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]

            # 获取过滤后的知识库
            search_text = self.search_var.get().lower()
            filtered_kbs = []

            for kb in self.knowledge_bases:
                if isinstance(kb, str):
                    if search_text in kb.lower():
                        filtered_kbs.append(kb)
                elif isinstance(kb, dict):
                    name = kb.get('name', '')
                    desc = kb.get('description', '')
                    if search_text in name.lower() or search_text in desc.lower():
                        filtered_kbs.append(kb)

            if index < len(filtered_kbs):
                kb = filtered_kbs[index]

                if isinstance(kb, str):
                    return kb
                elif isinstance(kb, dict):
                    return kb.get('id') or kb.get('name')

        return None

    def refresh_knowledge_bases(self):
        """刷新知识库列表（需要回调）"""
        if hasattr(self, 'on_refresh_callback'):
            self.on_refresh_callback()

class ModernRAGFlowGUI:
    """现代化RAGFlow GUI主程序"""

    def __init__(self):
        self.root = tk.Tk()

        # 先初始化颜色配置
        self.colors = ModernStyle.setup_theme()

        self.setup_window()

        # 应用实例
        self.app = None
        self.current_kb = None

        # 创建现代化界面
        self.create_widgets()

        # 状态变量
        self.is_connected = False
        self.is_processing = False

        # 配置知识库刷新回调
        if hasattr(self, 'kb_widget'):
            self.kb_widget.on_refresh_callback = self.refresh_knowledge_bases

    def setup_window(self):
        """设置现代化主窗口"""
        self.root.title("🚀 RAGFlow + LangChain 智能问答系统")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        # 设置窗口图标
        try:
            # 如果有图标文件可以取消注释
            # self.root.iconbitmap("icon.ico")
            pass
        except:
            pass

        # 应用现代化样式并保存颜色配置
        self.colors = ModernStyle.setup_theme()

        # 设置窗口背景色
        self.root.configure(bg=self.colors['bg'])

        # 设置窗口居中
        self.center_window()

    def center_window(self):
        """窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        """创建现代化界面组件"""
        # 创建主面板 - 使用Notebook实现标签页
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 标签页1: 问答界面
        chat_frame = ttk.Frame(notebook)
        notebook.add(chat_frame, text="💬 智能问答")

        # 标签页2: 配置界面
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="⚙️ 系统配置")

        # === 问答界面 ===
        self._create_chat_interface(chat_frame)

        # === 配置界面 ===
        self._create_config_interface(config_frame)

        # 底部状态栏
        self._create_status_bar()

    def _create_chat_interface(self, parent):
        """创建问答界面"""
        colors = ModernStyle.setup_theme()

        # 左侧面板 - 知识库选择
        left_frame = tk.Frame(parent, bg=colors['bg'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        left_frame.pack_propagate(False)
        left_frame.configure(width=300)

        # 连接控制
        connection_frame = ttk.LabelFrame(left_frame, text="🔗 连接控制", padding=10)
        connection_frame.pack(fill=tk.X, padx=10, pady=10)

        # 连接按钮容器
        button_container = tk.Frame(connection_frame)
        button_container.pack(fill=tk.X, pady=(0, 10))

        self.connect_button = ttk.Button(
            button_container,
            text="🔌 连接RAGFlow",
            command=self.toggle_connection,
            style='Primary.TButton',
            width=20
        )
        self.connect_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 状态指示器
        status_container = tk.Frame(connection_frame)
        status_container.pack(fill=tk.X)

        tk.Label(status_container, text="连接状态:", bg=colors['bg']).pack(side=tk.LEFT)

        self.status_indicator = tk.Label(
            status_container,
            text="🔴 未连接",
            bg=colors['bg'],
            fg=colors['danger'],
            font=('黑体', 9, 'bold')
        )
        self.status_indicator.pack(side=tk.LEFT, padx=(5, 0))

        # 知识库选择
        self.kb_widget = ModernKnowledgeBaseWidget(left_frame)
        self.kb_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 右侧面板 - 聊天区域
        right_frame = tk.Frame(parent, bg=colors['bg'])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 聊天标题
        title_frame = tk.Frame(right_frame, bg=colors['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(
            title_frame,
            text="💭 对话窗口",
            bg=colors['bg'],
            fg=colors['fg'],
            font=('黑体', 16, 'bold')
        )
        title_label.pack(side=tk.LEFT)

        # 聊天历史
        chat_container = tk.Frame(right_frame, bg=colors['bg'])
        chat_container.pack(fill=tk.BOTH, expand=True)

        self.chat_widget = ModernChatWidget(chat_container)
        self.chat_widget.pack(fill=tk.BOTH, expand=True)

        # 输入区域
        input_container = tk.Frame(right_frame, bg=colors['bg'])
        input_container.pack(fill=tk.X, pady=(10, 0))

        self.input_widget = ModernInputWidget(input_container, on_send_callback=self.send_question)
        self.input_widget.pack(fill=tk.X)

    def _create_config_interface(self, parent):
        """创建配置界面"""
        colors = ModernStyle.setup_theme()

        # 配置组件
        self.config_widget = ModernConfigWidget(parent)
        self.config_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_status_bar(self):
        """创建底部状态栏"""
        colors = ModernStyle.setup_theme()

        status_bar = tk.Frame(self.root, bg=colors['bg'], relief=tk.SUNKEN, bd=1)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 左侧状态信息
        left_status = tk.Frame(status_bar, bg=colors['bg'])
        left_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_label = tk.Label(
            left_status,
            text="🚀 RAGFlow智能问答系统就绪",
            bg=colors['bg'],
            fg=colors['fg'],
            font=('黑体', 8)
        )
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)

        # 右侧工具按钮
        right_tools = tk.Frame(status_bar, bg=colors['bg'])
        right_tools.pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            right_tools,
            text="🗑️ 清空",
            command=self.clear_chat,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            right_tools,
            text="💾 导出",
            command=self.export_chat,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            right_tools,
            text="ℹ️ 关于",
            command=self.show_about,
            width=8
        ).pack(side=tk.LEFT, padx=2)

    def toggle_connection(self):
        """切换连接状态"""
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        """连接到RAGFlow"""
        try:
            # 更新UI状态
            self.connect_button.config(text="⏳ 连接中...", state=tk.DISABLED)
            self.status_indicator.config(text="🟡 连接中...", fg=self.colors['warning'])
            self.status_label.config(text="🔄 正在连接到RAGFlow服务...")
            self.chat_widget.add_system_message("🔄 正在连接到RAGFlow服务，请稍候...")
            self.root.update()

            # 获取配置
            config = self.config_widget.get_config()

            # 创建应用实例
            self.app = RAGFlowLangChainApp(
                ragflow_url=config["ragflow_url"],
                ragflow_api_key=config["ragflow_api_key"],
                llm_model=config["llm_model"]
            )

            # 在后台线程中初始化
            threading.Thread(target=self._initialize_app, daemon=True).start()

        except Exception as e:
            self._on_connection_failed(f"连接失败: {e}")

    def _initialize_app(self):
        """后台初始化应用"""
        try:
            # 初始化应用
            if self.app.initialize():
                # 连接成功
                self.is_connected = True
                self.root.after(0, self._on_connection_success)
            else:
                # 连接失败
                self.root.after(0, self._on_connection_failed, "无法连接到RAGFlow服务")
        except Exception as e:
            self.root.after(0, self._on_connection_failed, str(e))

    def _on_connection_success(self):
        """连接成功处理"""
        self.connect_button.config(text="🔌 断开连接", state=tk.NORMAL, style='Danger.TButton')
        self.status_indicator.config(text="🟢 已连接", fg=self.colors['success'])
        self.status_label.config(text="✅ 连接成功！正在获取知识库列表...")
        self.chat_widget.add_system_message("✅ 连接成功！正在获取知识库列表...")

        # 启用输入
        self.input_widget.set_enabled(True)

        # 刷新知识库列表
        self.refresh_knowledge_bases()

    def _on_connection_failed(self, error_msg):
        """连接失败处理"""
        self.connect_button.config(text="🔌 连接RAGFlow", state=tk.NORMAL, style='Primary.TButton')
        self.status_indicator.config(text="🔴 连接失败", fg=self.colors['danger'])
        self.status_label.config(text=f"❌ {error_msg}")
        self.chat_widget.add_error_message(error_msg)

    def disconnect(self):
        """断开连接"""
        self.is_connected = False
        self.app = None
        self.current_kb = None

        self.connect_button.config(text="🔌 连接RAGFlow", state=tk.NORMAL, style='Primary.TButton')
        self.status_indicator.config(text="🔴 未连接", fg=self.colors['danger'])
        self.status_label.config(text="🔌 已断开连接")
        self.input_widget.set_enabled(False)

        self.chat_widget.add_system_message("🔌 已断开连接")

    def refresh_knowledge_bases(self):
        """刷新知识库列表"""
        if not self.is_connected or not self.app:
            return

        try:
            self.chat_widget.add_system_message("🔄 正在获取知识库列表...")
            knowledge_bases = self.app.connector.get_knowledge_bases()
            self.kb_widget.update_knowledge_bases(knowledge_bases)

            if knowledge_bases:
                self.chat_widget.add_system_message(f"✅ 发现 {len(knowledge_bases)} 个知识库，请选择一个开始对话")
            else:
                self.chat_widget.add_system_message("⚠️ 未发现任何知识库，请先在RAGFlow中创建知识库")

        except Exception as e:
            self.chat_widget.add_error_message(f"获取知识库列表失败: {e}")

    def send_question(self, question: str):
        """发送问题"""
        if self.is_processing:
            self.chat_widget.add_system_message("⏳ 请等待当前问题处理完成...")
            return

        if not self.is_connected or not self.app:
            self.chat_widget.add_error_message("❌ 请先连接到RAGFlow服务")
            return

        # 获取选中的知识库
        kb_name = self.kb_widget.get_selected_knowledge_base()
        if not kb_name:
            self.chat_widget.add_error_message("❌ 请先选择一个知识库")
            return

        # 添加用户消息
        self.chat_widget.add_user_message(question)

        # 更新状态
        self.is_processing = True
        self.input_widget.set_enabled(False)
        self.status_label.config(text="🤔 AI正在思考中...")

        # 显示处理状态
        self.chat_widget.add_system_message("🤔 AI正在分析您的问题，请稍候...")

        # 在后台线程中处理问题
        threading.Thread(
            target=self._process_question,
            args=(question, kb_name),
            daemon=True
        ).start()

    def _process_question(self, question: str, kb_name: str):
        """后台处理问题"""
        try:
            # 确保知识库已创建检索器
            if kb_name != self.current_kb:
                retriever = self.app.create_retriever(kb_name)
                if not retriever:
                    self.root.after(0, self._on_process_error, "无法创建知识库检索器")
                    return
                self.current_kb = kb_name

            # 获取配置
            config = self.config_widget.get_config()

            # 创建QA链
            qa_chain = self.app.create_qa_chain(kb_name, chain_type="with_sources")

            # 处理问题
            answer = qa_chain.invoke(question)

            # 返回结果
            self.root.after(0, self._on_process_success, answer)

        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))

    def _on_process_success(self, answer: str):
        """问题处理成功"""
        self.chat_widget.add_assistant_message(answer)
        self.is_processing = False
        self.input_widget.set_enabled(True)
        self.status_label.config(text="✅ 就绪")

    def _on_process_error(self, error_msg: str):
        """问题处理失败"""
        self.chat_widget.add_error_message(f"❌ 回答问题时出错: {error_msg}")
        self.is_processing = False
        self.input_widget.set_enabled(True)
        self.status_label.config(text="⚠️ 处理完成（有错误）")

    def clear_chat(self):
        """清空对话"""
        if messagebox.askyesno("🗑️ 确认", "确定要清空所有对话记录吗？"):
            self.chat_widget.clear_history()

    def export_chat(self):
        """导出对话记录"""
        try:
            # 选择保存文件
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[
                    ("文本文件", "*.txt"),
                    ("Markdown文件", "*.md"),
                    ("所有文件", "*.*")
                ],
                title="保存对话记录"
            )

            if filename:
                # 获取消息内容
                messages = self.chat_widget.messages

                with open(filename, 'w', encoding='utf-8') as f:
                    if filename.endswith('.md'):
                        # Markdown格式
                        f.write("# RAGFlow智能问答系统 - 对话记录\n\n")
                        f.write(f"**导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write("---\n\n")

                        for message in messages:
                            if message.message_type == "user":
                                f.write(f"## 👤 **您** ({message.timestamp})\n")
                                f.write(f"{message.content}\n\n")
                            elif message.message_type == "assistant":
                                f.write(f"## 🤖 **AI助手** ({message.timestamp})\n")
                                f.write(f"{message.content}\n\n")
                            elif message.message_type == "error":
                                f.write(f"## ❌ **错误** ({message.timestamp})\n")
                                f.write(f"{message.content}\n\n")
                            else:  # system
                                f.write(f"ℹ️ **{message.timestamp}**: {message.content}\n\n")
                                f.write("---\n\n")
                    else:
                        # 文本格式
                        f.write("RAGFlow智能问答系统 - 对话记录\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write("-" * 50 + "\n\n")

                        for message in messages:
                            f.write(f"[{message.timestamp}] ")

                            if message.message_type == "user":
                                f.write("您: ")
                            elif message.message_type == "assistant":
                                f.write("AI助手: ")
                            elif message.message_type == "error":
                                f.write("错误: ")
                            else:
                                f.write("系统: ")

                            f.write(f"{message.content}\n\n")

                messagebox.showinfo("✅ 成功", f"对话记录已导出到: {filename}")

        except Exception as e:
            messagebox.showerror("❌ 错误", f"导出失败: {e}")

    def show_about(self):
        """显示关于信息"""
        about_text = """🚀 RAGFlow + LangChain 智能问答系统 v2.0

基于RAGFlow知识库和LangChain框架构建的现代化智能问答系统。

✨ 主要功能：
• 🔗 现代化RAGFlow连接管理
• 📚 智能知识库选择和搜索
• 💬 实时问答对话体验
• 🎨 现代化用户界面设计
• 💾 对话记录管理
• ⚙️ 灵活的配置管理

🛠️ 技术栈：
• Python 3.8+
• Tkinter (现代化GUI)
• RAGFlow (知识库)
• LangChain (AI框架)
• OpenAI/GLM API

🎯 界面特性：
• 现代化设计风格
• 响应式布局
• 实时状态更新
• 智能输入提示
• 多格式导出

👨‍💻 开发者: SUSU
📅 更新时间: 2025
📧 版本: 2.0 - 现代化版本"""

        messagebox.showinfo("ℹ️ 关于", about_text)

    def run(self):
        """运行GUI应用"""
        self.root.mainloop()

def main():
    """主函数"""
    try:
        app = ModernRAGFlowGUI()
        app.run()
    except Exception as e:
        messagebox.showerror("❌ 启动错误", f"应用启动失败: {e}")

if __name__ == "__main__":
    main()