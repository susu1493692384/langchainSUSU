#!/usr/bin/env python3
"""
高级工具示例 - 展示如何扩展和修改工具
"""

from langchain_core.tools import tool
from datetime import datetime, timedelta
import json
import random
import hashlib

# 1. 增强版时间工具
@tool
def get_time_info(location: str = "本地", format: str = "标准") -> str:
    """获取指定地区的时间信息

    Args:
        location: 地区名称（如：北京、纽约、伦敦）
        format: 时间格式（标准、详细、ISO）
    """
    try:
        now = datetime.now()

        if format == "详细":
            time_str = now.strftime(f"{location}时间：%Y年%m月%d日 %H:%M:%S 星期%A")
        elif format == "ISO":
            time_str = now.isoformat()
        else:
            time_str = f"{location}时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"

        return time_str
    except Exception as e:
        return f"获取时间失败：{str(e)}"

# 2. 高级计算工具
@tool
def advanced_calculator(expression: str, operation: str = "calculate") -> str:
    """高级数学计算工具

    Args:
        expression: 数学表达式或数值
        operation: 操作类型（calculate、sqrt、factorial、prime_check）
    """
    try:
        if operation == "calculate":
            # 安全的数学计算
            allowed_chars = set("0123456789+-*/().")
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"计算结果：{expression} = {result}"
            else:
                return f"表达式包含不支持的字符：{expression}"

        elif operation == "sqrt":
            num = float(expression)
            result = num ** 0.5
            return f"√{num} = {result}"

        elif operation == "factorial":
            num = int(expression)
            if num < 0:
                return "负数没有阶乘"
            result = 1
            for i in range(1, num + 1):
                result *= i
            return f"{num}! = {result}"

        elif operation == "prime_check":
            num = int(expression)
            if num < 2:
                return f"{num} 不是质数"
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return f"{num} 不是质数（可被 {i} 整除）"
            return f"{num} 是质数"

        else:
            return f"不支持的操作：{operation}"

    except Exception as e:
        return f"计算错误：{str(e)}"

# 3. 文件处理工具
@tool
def file_manager(operation: str, filename: str, content: str = "") -> str:
    """文件管理工具

    Args:
        operation: 操作类型（read、write、append、delete、exists）
        filename: 文件名
        content: 文件内容（用于write和append操作）
    """
    try:
        if operation == "read":
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"文件 '{filename}' 内容：\n{content}"

        elif operation == "write":
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"成功写入文件：{filename}"

        elif operation == "append":
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"\n{content}")
            return f"成功追加内容到文件：{filename}"

        elif operation == "exists":
            import os
            exists = os.path.exists(filename)
            return f"文件 '{filename}' {'存在' if exists else '不存在'}"

        elif operation == "delete":
            import os
            if os.path.exists(filename):
                os.remove(filename)
                return f"成功删除文件：{filename}"
            else:
                return f"文件不存在：{filename}"

        else:
            return f"不支持的操作：{operation}"

    except Exception as e:
        return f"文件操作失败：{str(e)}"

# 4. 随机数生成工具
@tool
def random_generator(type: str, min_val: int = 1, max_val: int = 100, count: int = 1) -> str:
    """随机数生成工具

    Args:
        type: 类型（integer、float、string、choice）
        min_val: 最小值（用于integer和float）
        max_val: 最大值（用于integer和float）
        count: 生成数量
    """
    try:
        if type == "integer":
            numbers = [random.randint(min_val, max_val) for _ in range(count)]
            return f"随机整数：{numbers}"

        elif type == "float":
            numbers = [round(random.uniform(min_val, max_val), 2) for _ in range(count)]
            return f"随机浮点数：{numbers}"

        elif type == "string":
            import string
            length = min_val if min_val > 0 else 8
            chars = string.ascii_letters + string.digits
            result = [''.join(random.choices(chars, k=length)) for _ in range(count)]
            return f"随机字符串：{result}"

        elif type == "choice":
            # 需要提供选项，这里使用默认选项
            options = ["选项A", "选项B", "选项C", "选项D"]
            selected = random.choice(options)
            return f"随机选择：{selected}"

        else:
            return f"不支持的类型：{type}"

    except Exception as e:
        return f"生成随机数失败：{str(e)}"

# 5. 数据转换工具
@tool
def data_converter(data: str, from_format: str, to_format: str) -> str:
    """数据格式转换工具

    Args:
        data: 要转换的数据
        from_format: 输入格式（json、text、base64、url）
        to_format: 输出格式（json、text、base64、url）
    """
    try:
        if from_format == "json" and to_format == "text":
            parsed = json.loads(data)
            return f"JSON转文本：{json.dumps(parsed, indent=2, ensure_ascii=False)}"

        elif from_format == "text" and to_format == "json":
            # 简单的文本转JSON
            return f"文本转JSON：{json.dumps({'text': data}, ensure_ascii=False)}"

        elif from_format == "base64" and to_format == "text":
            import base64
            decoded = base64.b64decode(data).decode('utf-8')
            return f"Base64解码：{decoded}"

        elif from_format == "text" and to_format == "base64":
            import base64
            encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
            return f"Base64编码：{encoded}"

        elif from_format == "text" and to_format == "url":
            from urllib.parse import quote
            encoded = quote(data)
            return f"URL编码：{encoded}"

        elif from_format == "url" and to_format == "text":
            from urllib.parse import unquote
            decoded = unquote(data)
            return f"URL解码：{decoded}"

        else:
            return f"不支持的转换：{from_format} -> {to_format}"

    except Exception as e:
        return f"转换失败：{str(e)}"

def demonstrate_advanced_tools():
    """演示高级工具的使用"""
    print("=== 高级工具演示 ===\n")

    # 1. 增强时间工具
    print("1. 增强时间工具:")
    print(get_time_info.invoke({"location": "北京", "format": "详细"}))
    print(get_time_info.invoke({"location": "纽约", "format": "ISO"}))
    print()

    # 2. 高级计算器
    print("2. 高级计算器:")
    print(advanced_calculator.invoke({"expression": "2+3*4", "operation": "calculate"}))
    print(advanced_calculator.invoke({"expression": "144", "operation": "sqrt"}))
    print(advanced_calculator.invoke({"expression": "5", "operation": "factorial"}))
    print(advanced_calculator.invoke({"expression": "17", "operation": "prime_check"}))
    print()

    # 3. 随机数生成器
    print("3. 随机数生成器:")
    print(random_generator.invoke({"type": "integer", "min_val": 1, "max_val": 10, "count": 3}))
    print(random_generator.invoke({"type": "float", "min_val": 0, "max_val": 1, "count": 2}))
    print(random_generator.invoke({"type": "string", "min_val": 6, "count": 2}))
    print(random_generator.invoke({"type": "choice"}))
    print()

    # 4. 数据转换
    print("4. 数据转换工具:")
    test_text = "Hello LangChain"
    base64_result = data_converter.invoke({"data": test_text, "from_format": "text", "to_format": "base64"})
    print(base64_result)

    # 提取base64结果进行解码
    import re
    base64_match = re.search(r'Base64编码：([A-Za-z0-9+/=]+)', base64_result)
    if base64_match:
        encoded = base64_match.group(1)
        decoded = data_converter.invoke({"data": encoded, "from_format": "base64", "to_format": "text"})
        print(decoded)
    print()

def main():
    print("LangChain 高级工具扩展示例\n")

    try:
        demonstrate_advanced_tools()

        print("=== 工具扩展要点 ===")
        print("1. 添加更多参数增加工具灵活性")
        print("2. 提供多种操作模式")
        print("3. 完善错误处理和边界检查")
        print("4. 添加详细的参数说明")
        print("5. 支持链式操作和复杂逻辑")

    except Exception as e:
        print(f"演示过程中出错: {e}")

if __name__ == "__main__":
    main()