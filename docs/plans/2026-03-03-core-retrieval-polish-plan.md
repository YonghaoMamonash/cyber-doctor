# Cyber-Doctor Core Retrieval Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 完成一期核心链路优化（离线索引 + 在线检索 + 生成质量 + 评估闭环），在现有项目上落地可配置、可扩展、可验证的 RAG/KG 优化体系，并为二期 Agent 能力预留接口。

**Architecture:** 维持当前单体应用结构（Gradio + QA function_tool 编排），引入“配置驱动检索策略层”。核心变更集中在 `model/RAG`、`rag`、`qa/function_tool.py`、`config/config-web.yaml`，通过新增查询改写与多查询融合、Embedding 双引擎、KG 关系过滤、评估脚本与基础可观测能力实现质量提升。

**Tech Stack:** Python 3.10+, LangChain/LangChain-Community, FAISS, ModelScopeEmbeddings, ZhipuAIEmbeddings, py2neo, Gradio

---

## 1. Scope and Success Criteria

### In Scope (一期)
- 离线数据准备优化
  - 文档元数据增强（来源、更新时间）
  - 分块策略参数化（递归切分可配置）
  - Embedding 双引擎（ModelScope / Zhipu embedding-3）可切换
  - 过期文档过滤策略（时间敏感数据）
- 在线应用优化
  - 问题改写（基于历史对话生成独立问题）
  - Multi-Query 多视角检索 + 去重融合
  - KG 关系类型过滤（减少知识图谱噪音）
  - 检索失败降级与错误处理
- 生成环节优化
  - 增加轻量 self-check 机制（回答前依据检索内容进行一致性约束）
- 评估闭环
  - 提供离线评估数据模板与运行脚本（RAGAS 兼容）

### Out of Scope (二期)
- 完整 RAPTOR 树索引工程化实现
- 完整 Self-RAG 反思-重检索-重生成多轮图
- 全量 Agent 记忆系统与 A2A 多 Agent 编排

### Success Criteria
- 配置切换 Embedding 引擎后可完成索引构建与检索
- RAG 查询支持独立问题改写 + 多查询融合
- KG 查询只注入允许关系类型的图谱信息
- 新增评估脚本可对样例数据输出指标报告
- 不破坏现有文本/图片/音视频/PPT/Docx主流程

## 2. Public Interfaces / Config Changes

### Config additions (`config/config-web.yaml`)
- `model.embedding.provider`: `modelscope | zhipuai`
- `model.embedding.zhipu-model`: default `embedding-3`
- `model.embedding.api-key-env`: default `EMBEDDING_API_KEY`
- `model.rag.indexing.chunk-size`
- `model.rag.indexing.chunk-overlap`
- `model.rag.indexing.stale-days` (`0` 为不过滤)
- `model.rag.retrieval.top-k-per-query`
- `model.rag.retrieval.max-context-docs`
- `model.rag.query-rewrite.enabled`
- `model.rag.multi-query.enabled`
- `model.rag.multi-query.count`
- `model.rag.answer-self-check.enabled`

### Env additions (`.env.example`)
- `EMBEDDING_API_KEY` (optional; fallback 到 `LLM_API_KEY`)

### New modules
- `rag/query_optimizer.py`: 问题改写 + 多查询生成
- `rag/retrieval_fusion.py`: 多查询检索去重融合
- `evaluation/rag_eval_runner.py`: 评估脚本
- `evaluation/sample_dataset.jsonl`: 样例评估集模板

## 3. Implementation Tasks

### Task 1: 配置驱动的索引与检索参数化
**Files:**
- Modify: `config/config-web.yaml`
- Modify: `.env.example`
- Modify: `model/RAG/retrieve_model.py`
- Modify: `model/Internet/Internet_model.py`

**Steps:**
1. 新增 RAG/Embedding 相关配置项与默认值。
2. 在 `retrieve_model.py` 中抽象 Embedding 构建器：按 `provider` 选择 ModelScope 或 Zhipu。
3. 将 `chunk_size/chunk_overlap` 改为读取配置。
4. 在文档加载后补充 metadata（source、last_modified_ts）。
5. 添加 stale-days 过滤逻辑。
6. 在互联网检索模型中统一 embedding 配置来源（保持同策略）。

### Task 2: 在线问题改写与多查询融合检索
**Files:**
- Add: `rag/query_optimizer.py`
- Add: `rag/retrieval_fusion.py`
- Modify: `rag/retrieve/retrieve_document.py`
- Modify: `rag/rag_chain.py`

**Steps:**
1. 实现 `rewrite_question(question, history)`：将上下文问题改写为独立问题。
2. 实现 `generate_multi_queries(question, n)`：生成多视角子查询。
3. 实现检索融合：每个子查询检索 top-k，按 `(source, content_hash)` 去重，截断到 `max-context-docs`。
4. `rag_chain.invoke` 改为使用“改写 -> 多查询 -> 融合上下文 -> 生成”的流程。
5. 保留失败回退：任一环节异常时降级为原单查询流程。

### Task 3: KG 关系类型过滤与注入降噪
**Files:**
- Modify: `qa/function_tool.py`
- Modify: `config/config-web.yaml`

**Steps:**
1. 读取 `database.neo4j.relationship-type` 白名单。
2. 在 `relation_tool` 仅保留允许关系类型。
3. 添加实体/关系上限，避免超长提示污染主回答。
4. 未命中或异常时不注入 KG 信息，直接走普通回答。

### Task 4: 生成阶段轻量 self-check
**Files:**
- Modify: `rag/rag_chain.py`

**Steps:**
1. 增加可配置 self-check prompt（仅在 `answer-self-check.enabled=true` 生效）。
2. 先生成初稿，再让模型基于上下文做一次一致性检查并输出最终版本。
3. 失败自动回退到初稿，避免影响可用性。

### Task 5: 评估脚本（RAGAS 兼容）
**Files:**
- Add: `evaluation/rag_eval_runner.py`
- Add: `evaluation/sample_dataset.jsonl`
- Modify: `requirements.txt` (新增可选评估依赖)

**Steps:**
1. 定义样例数据格式：`question`, `ground_truth`, `contexts`, `answer`。
2. 脚本支持本地跑一批 query 并保存结果。
3. 如安装 ragas，则输出核心指标；未安装则输出基础统计与提示。

### Task 6: 基础可观测与鲁棒性
**Files:**
- Modify: `Internet/Internet_chain.py`
- Modify: `model/RAG/retrieve_service.py`

**Steps:**
1. 为联网抓取增加线程安全字典写入与文件名清洗。
2. 统一错误日志输出格式，保留关键上下文。
3. 修复用户 retriever 为空时的空指针风险。

## 4. Test Cases and Validation

### Unit-level
- `rewrite_question` 在有历史和无历史两种场景的输出合法性。
- `generate_multi_queries` 的去重与最少回退（至少包含原问题）。
- `retrieval_fusion` 去重、截断、空检索兼容。
- `relation_tool` 对白名单关系过滤是否生效。
- `stale-days` 过滤是否剔除过期文档。

### Integration-level
- 文本问答（普通）回归：不应受影响。
- “根据知识库...” 问答：应走改写+多查询检索。
- “根据知识图谱...” 问答：回答包含受控 KG 信息。
- 联网搜索：并发抓取无路径异常，失败可回退。

### Manual scenarios
- 上传 PDF + 追问（上下文依赖）：独立问题改写应更稳定。
- 时间敏感问题（新旧版本文档共存）：过期过滤应生效。
- 异常网络/异常 API Key：系统应降级而非崩溃。

## 5. Risks and Mitigations

- 风险: LLM 生成多查询格式不稳定。
  - 缓解: 强约束 JSON + 容错解析 + 回退原问题。
- 风险: 云端 embedding 失败导致索引不可用。
  - 缓解: provider 回退到本地模型。
- 风险: self-check 增加延迟。
  - 缓解: 可配置开关，默认可关闭。

## 6. Assumptions and Defaults

- 默认按一期范围执行，不在本轮实现 A2A 多 Agent。
- 默认采用 Embedding 双引擎可切换策略，默认值 `modelscope`。
- 默认继续使用 FAISS 本地向量库，不引入额外外部向量数据库。
- 默认保持现有 Gradio 接口和函数路由结构，避免大规模重构。

## 7. Execution Order (This Session)

1. Task 1 配置与 embedding 双引擎。
2. Task 2 查询改写 + 多查询融合检索。
3. Task 3 KG 关系过滤。
4. Task 6 联网线程安全与检索空指针修复。
5. 运行静态检查/基础回归（import + 关键函数 smoke test）。
6. 输出变更说明与下一批（Task 4/Task 5）执行建议。
