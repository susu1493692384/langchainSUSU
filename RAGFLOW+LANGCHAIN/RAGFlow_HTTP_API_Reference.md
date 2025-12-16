# RAGFlow HTTP API 参考文档

## 📋 目录

1. [认证方式](#认证方式)
2. [数据集管理](#数据集管理)
   - [创建数据集](#创建数据集)
   - [列出数据集](#列出数据集)
   - [更新数据集](#更新数据集)
   - [删除数据集](#删除数据集)
3. [响应格式](#响应格式)
4. [错误代码](#错误代码)

---

## 🔐 认证方式

所有 API 请求都需要在请求头中包含有效的 API 密钥：

```http
Authorization: Bearer <YOUR_API_KEY>
Content-Type: application/json
```

---

## 📚 数据集管理

### 创建数据集

**端点：** `POST /api/v1/datasets`

创建一个新的数据集（知识库）。

#### 请求参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `name` | string | ✅ | 数据集的唯一名称（最多128字符） |
| `avatar` | string | ❌ | Base64编码的头像图片（最多65535字符） |
| `description` | string | ❌ | 数据集描述（最多65535字符） |
| `embedding_model` | string | ❌ | 嵌入模型名称，格式：`model_name@model_factory` |
| `permission` | string | ❌ | 权限设置：`me`(默认) 或 `team` |
| `chunk_method` | string | ❌ | 分块方式（见下表） |
| `parser_config` | object | ❌ | 解析器配置 |
| `parse_type` | int | ❌ | 数据摄入流程中的解析器类型ID |
| `pipeline_id` | string | ❌ | 数据摄入流程的32位十六进制ID |

#### 分块方式选项

| 值 | 描述 |
|----|------|
| `naive` | 普通模式（默认） |
| `book` | 书籍 |
| `email` | 电子邮件 |
| `laws` | 法律 |
| `manual` | 手动操作 |
| `one` | 第一个 |
| `paper` | 论文 |
| `picture` | 图片 |
| `presentation` | 演示文稿 |
| `qa` | 问答环节 |
| `table` | 表格 |
| `tag` | 标签 |

#### 请求示例

**基础创建：**
```bash
curl --request POST \
     --url http://localhost:9380/api/v1/datasets \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer YOUR_API_KEY' \
     --data '{
       "name": "test_dataset",
       "description": "测试数据集"
     }'
```

**完整配置创建：**
```bash
curl --request POST \
     --url http://localhost:9380/api/v1/datasets \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer YOUR_API_KEY' \
     --data '{
       "name": "complete_dataset",
       "description": "完整配置的测试数据集",
       "embedding_model": "BAAI/bge-large-zh-v1.5@BAAI",
       "permission": "me",
       "chunk_method": "naive",
       "parser_config": {
         "chunk_token_num": 512,
         "delimiter": "\\n!?;。；！？",
         "auto_keywords": 0,
         "auto_questions": 0,
         "html4excel": false,
         "layout_recognize": "DeepDOC",
         "raptor": {"use_raptor": false},
         "graphrag": {"use_graphrag": false}
       }
     }'
```

#### 使用数据摄入管道创建：

```bash
curl --request POST \
  --url http://localhost:9380/api/v1/datasets \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer YOUR_API_KEY' \
  --data '{
   "name": "pipeline_dataset",
   "parse_type": 1,
   "pipeline_id": "d0bebe30ae2211f0970942010a8e0005"
  }'
```

---

### 列出数据集

**端点：** `GET /api/v1/datasets`

获取所有数据集列表。

#### 查询参数

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `page` | int | ❌ | 1 | 页码 |
| `page_size` | int | ❌ | undefined | 每页显示数量（最大30） |
| `orderby` | string | ❌ | create_time | 排序字段：`create_time` 或 `update_time` |
| `desc` | boolean | ❌ | true | 是否降序排列 |
| `name` | string | ❌ | - | 按名称过滤 |
| `id` | string | ❌ | - | 按ID过滤 |

#### 请求示例

```bash
curl --request GET \
     --url "http://localhost:9380/api/v1/datasets?page=1&page_size=10&orderby=create_time&desc=true" \
     --header 'Authorization: Bearer YOUR_API_KEY'
```

---

### 更新数据集

**端点：** `PUT /api/v1/datasets/{dataset_id}`

更新指定数据集的配置信息。

#### 路径参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `dataset_id` | string | ✅ | 要更新的数据集ID |

#### 请求参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `name` | string | ❌ | 新的数据集名称 |
| `avatar` | string | ❌ | 新的Base64编码头像 |
| `description` | string | ❌ | 新的描述 |
| `embedding_model` | string | ❌ | 新的嵌入模型 |
| `permission` | string | ❌ | 新的权限设置 |
| `chunk_method` | string | ❌ | 新的分块方式 |
| `pagerank` | int | ❌ | 页面排名（默认0） |
| `parser_config` | object | ❌ | 新的解析器配置 |

#### 请求示例

```bash
curl --request PUT \
     --url "http://localhost:9380/api/v1/datasets/your_dataset_id" \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer YOUR_API_KEY' \
     --data '{
       "name": "updated_dataset_name",
       "description": "更新后的描述",
       "pagerank": 1
     }'
```

---

### 删除数据集

**端点：** `DELETE /api/v1/datasets`

删除一个或多个数据集。

#### 请求参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `ids` | list[string] | ✅ | 要删除的数据集ID列表，`null` 表示删除所有 |

#### 请求示例

**删除指定数据集：**
```bash
curl --request DELETE \
     --url http://localhost:9380/api/v1/datasets \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer YOUR_API_KEY' \
     --data '{
       "ids": ["d94a8dc02c9711f0930f7fbc369eab6d", "e94a8dc02c9711f0930f7fbc369eab6e"]
     }'
```

**删除所有数据集：**
```bash
curl --request DELETE \
     --url http://localhost:9380/api/v1/datasets \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer YOUR_API_KEY' \
     --data '{
       "ids": null
     }'
```

---

## 📊 响应格式

### 成功响应

所有成功的API调用都返回以下格式：

```json
{
    "code": 0,
    "data": {
        // 具体的响应数据
    }
}
```

### 数据集对象结构

```json
{
    "avatar": "Base64编码的头像",
    "chunk_count": 59,
    "chunk_method": "naive",
    "create_date": "Sat, 14 Sep 2024 01:12:37 GMT",
    "create_time": 1726276357324,
    "created_by": "69736c5e723611efb51b0242ac120007",
    "description": "数据集描述",
    "document_count": 1,
    "embedding_model": "BAAI/bge-large-zh-v1.5@BAAI",
    "id": "6e211ee0723611efa10a0242ac120007",
    "language": "English",
    "name": "数据集名称",
    "pagerank": 0,
    "parser_config": {
        "chunk_token_num": 8192,
        "delimiter": "\\n",
        "auto_keywords": 0,
        "auto_questions": 0,
        "html4excel": false,
        "layout_recognize": "DeepDOC",
        "raptor": {
            "use_raptor": false
        },
        "graphrag": {
            "use_graphrag": false
        }
    },
    "permission": "me",
    "similarity_threshold": 0.2,
    "status": "1",
    "tenant_id": "69736c5e723611efb51b0242ac120007",
    "token_num": 12744,
    "update_date": "Thu, 10 Oct 2024 04:07:23 GMT",
    "update_time": 1728533243536,
    "vector_similarity_weight": 0.3
}
```

---

## ❌ 错误代码

| 错误代码 | 描述 | 示例消息 |
|----------|------|----------|
| 0 | 成功 | "Success" |
| 101 | 数据集名称已存在 | "Dataset name 'test' already exists" |
| 102 | 权限错误 | "You don't own the dataset." |
| 102 | 数据集不存在 | "The dataset doesn't exist" |
| 102 | 修改被禁止 | "Can't change tenant_id." |

### 错误响应格式

```json
{
    "code": 101,
    "message": "Dataset name 'test' already exists"
}
```

---

## 🔧 解析器配置详解

### Naive 分块方式配置

当 `chunk_method` 为 `"naive"` 时，`parser_config` 支持以下参数：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `auto_keywords` | int | 0 | 自动提取关键词数量 |
| `auto_questions` | int | 0 | 自动生成问题数量 |
| `chunk_token_num` | int | 512 | 每个分块的token数量 |
| `delimiter` | string | "\\n" | 分隔符 |
| `html4excel` | boolean | false | 是否将Excel转换为HTML |
| `layout_recognize` | string | "DeepDOC" | 布局识别方式 |
| `tag_kb_ids` | array[string] | - | 标签分块法的数据集ID列表 |
| `task_page_size` | int | 12 | PDF文件的页面大小 |
| `raptor` | object | {"use_raptor": false} | RAPTOR相关设置 |
| `graphrag` | object | {"use_graphrag": false} | GraphRAG相关设置 |

### 其他分块方式配置

当 `chunk_method` 为以下值时，`parser_config` 仅需包含：
- `"qa"`, `"manual"`, `"paper"`, `"book"`, `"laws"`, `"presentation"`：只需 `raptor` 配置
- `"table"`, `"picture"`, `"one"`, `"email"`：空的JSON对象

---

## 🚀 快速开始示例

### Python 示例

```python
import requests
import json

# 配置
BASE_URL = "http://localhost:9380"
API_KEY = "YOUR_API_KEY"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 创建数据集
def create_dataset():
    url = f"{BASE_URL}/api/v1/datasets"
    data = {
        "name": "my_test_dataset",
        "description": "我的测试数据集",
        "chunk_method": "naive",
        "parser_config": {
            "chunk_token_num": 512,
            "delimiter": "\\n"
        }
    }

    response = requests.post(url, headers=HEADERS, json=data)
    return response.json()

# 列出数据集
def list_datasets():
    url = f"{BASE_URL}/api/v1/datasets"
    response = requests.get(url, headers=HEADERS)
    return response.json()

# 删除数据集
def delete_dataset(dataset_id):
    url = f"{BASE_URL}/api/v1/datasets"
    data = {"ids": [dataset_id]}
    response = requests.delete(url, headers=HEADERS, json=data)
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 创建数据集
    result = create_dataset()
    print("创建结果:", result)

    if result.get("code") == 0:
        dataset_id = result["data"]["id"]
        print(f"数据集创建成功，ID: {dataset_id}")

        # 列出所有数据集
        datasets = list_datasets()
        print("数据集列表:", datasets)

        # 删除数据集
        delete_result = delete_dataset(dataset_id)
        print("删除结果:", delete_result)
```

---

## 📝 注意事项

1. **端口配置**：RAGFlow API 默认使用 `9380` 端口，不是 Web UI 的 `9000` 端口
2. **互斥参数**：`chunk_method` 与 `parse_type`/`pipeline_id` 互斥，只能选择其中一种方式
3. **默认行为**：如果未指定分块方式，系统默认使用 `chunk_method = "naive"`
4. **权限管理**：只有数据集的创建者或团队成员（根据 `permission` 设置）才能修改或删除数据集
5. **ID格式**：`pipeline_id` 必须是32位小写十六进制字符串
6. **编码限制**：数据集名称仅支持 BMPF 基本多文种平面格式

---

## 🔗 相关资源

- [RAGFlow 官方文档](https://ragflow.io/docs)
- [RAGFlow GitHub 仓库](https://github.com/infiniflow/ragflow)
- [API 完整参考](https://ragflow.io/docs/dev/http_api_reference)

---

*最后更新：2025年1月*