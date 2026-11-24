# Solar PV RAG Agent

AI Agent 驱动的太阳能光伏材料**检索增强推理**系统。基于 RAG（Retrieval-Augmented Generation）构建光伏材料领域知识库问答，并通过可调用多工具的 Agent 完成「自然语言 → 工具调用 → 推理回答」，最后用 RAGAS 对生成质量做量化评测。

> 本项目既可作为毕业设计（太阳能光伏材料方向 × AI）的工程底座，也可作为简历中的 AI 工程实践项目。

## ✨ 功能特性

- **领域知识库问答（RAG）**：PDF / TXT / Markdown 文献入库，混合检索（FAISS 向量 + BM25 稀疏）召回，LLM 生成带来源的回答。
- **多工具 AI Agent**：Agent 可自主调用四类工具——
  - `literature_search`：光伏文献知识库检索；
  - `material_database`：材料数据库查询（本地示例库 + 可选 Materials Project）；
  - `sq_efficiency`：Shockley-Queisser 单结理论极限效率估算；
  - `bandgap_to_wavelength`：带隙与吸收截止波长换算。
- **RAGAS 量化评测**：对 faithfulness（忠实度）、answer relevancy（答案相关性）、context precision / recall（上下文精确率 / 召回率）做评估。
- **开箱即用**：Embedding 默认本地 BGE 模型，无需 API key 即可构建索引；LLM 支持 OpenAI 兼容接口（OpenAI / DeepSeek / Qwen）。

## 🏗 架构

```
                         ┌──────────────────────────────┐
                         │          用户提问            │
                         └──────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │      AI Agent（LLM + 工具）    │
                        │  ReAct / tool-calling 循环     │
                        └──┬──────────┬──────────┬───────┘
                文献检索    │   材料库  │  物性计算│
                           ▼          ▼          ▼
              ┌────────────────┐ ┌─────────┐ ┌──────────┐
              │ HybridRetriever │ │materials│ │ physics  │
              │ FAISS + BM25    │ │  .csv   │ │ S-Q/波长 │
              └───────┬────────┘ └─────────┘ └──────────┘
                      ▼
              ┌────────────────┐
              │  文献知识库     │  ← data/raw_pdfs（构建索引）
              │  FAISS + BM25  │
              └────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 建议 Python 3.10 / 3.11
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，至少配置 LLM（可接 DeepSeek / Qwen / OpenAI 兼容接口）：

```ini
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com/v1   # 示例：DeepSeek
LLM_MODEL=deepseek-chat
```

Embedding 默认本地 `BAAI/bge-small-zh-v1.5`，无需额外配置。

### 3. 放入文献并构建索引

把文献（PDF / TXT / MD）放入 `data/raw_pdfs/`：

```bash
# 示例：把自带的演示文档复制进去试跑
cp data/sample_docs/solar_pv_overview.md data/raw_pdfs/

python scripts/build_index.py
```

### 4. 运行 Agent

```bash
python scripts/run_agent.py
```

示例提问：

- 「钙钛矿太阳能电池有哪些优势和劣势？」
- 「MAPbI3 的带隙是多少，对应理论极限效率多高？」
- 「带隙 1.5 eV 的材料，其吸收截止波长是多少？」

### 5. RAGAS 评测（可选）

```bash
pip install -r requirements-eval.txt
python scripts/evaluate.py
```

评测集会读取 `data/eval_set.json`，自动跑一遍「检索 + 生成」后输出四项指标，并保存到 `data/eval_result.json`。

## 📁 目录结构

```
solar-pv-rag-agent/
├── src/solar_rag/            # 核心包
│   ├── config.py             # 配置（.env 加载）
│   ├── loader.py             # 文档加载（PDF/TXT/MD）
│   ├── splitter.py           # 文本切分
│   ├── embeddings.py         # Embedding 后端（本地 BGE / OpenAI）
│   ├── index.py              # FAISS + BM25 索引构建与加载
│   ├── retrieval.py          # 混合检索（向量 + BM25 加权融合）
│   ├── llm.py                # LLM 后端（OpenAI 兼容）
│   ├── agent.py              # 多工具 Agent 装配
│   ├── evaluation.py         # RAGAS 评测封装
│   └── tools/                # Agent 工具集
│       ├── literature.py     # 文献检索工具
│       ├── materials.py      # 材料数据库工具
│       └── physics.py        # 物性计算工具
├── scripts/
│   ├── build_index.py        # 构建索引
│   ├── run_agent.py          # 交互式运行 Agent
│   └── evaluate.py           # RAGAS 评测
├── data/
│   ├── sample_docs/          # 示例文献
│   ├── materials.csv         # 本地示例材料库
│   └── eval_set.json         # 示例评测集
├── requirements.txt
├── requirements-eval.txt
├── pyproject.toml
└── .env.example
```

## 🧪 评测指标（RAGAS）

| 指标 | 含义 |
|------|------|
| faithfulness | 回答是否忠实于检索到的上下文（有无幻觉） |
| answer_relevancy | 回答与问题的相关程度 |
| context_precision | 检索上下文的精确率 |
| context_recall | 检索上下文的召回率 |

## 🛠 技术栈

- **LLM / Agent**：LangChain（`create_tool_calling_agent` + `AgentExecutor`）
- **向量检索**：FAISS
- **稀疏检索**：BM25（rank-bm25 + jieba 分词）
- **Embedding**：sentence-transformers（BGE）/ OpenAI
- **评测**：RAGAS
- **材料数据**：本地 CSV + Materials Project（可选）

## ⚖ License

MIT License © 2026 张震寰

## ⚠️ 说明

- 仓库不包含任何文献 PDF 与索引文件（已在 `.gitignore` 排除），请使用你自己的文献构建知识库；
- 请勿在 `.env` 或代码中提交真实的 API key；
- 示例材料数据为演示用近似值，正式科研请以文献或数据库实测为准。
