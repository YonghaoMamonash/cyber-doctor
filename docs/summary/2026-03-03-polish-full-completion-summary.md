# Polish 全量收尾总结（按原始思考顺序）

日期：2026-03-03

## 一、离线数据准备阶段

### 1）数据提取：高信噪比与时效处理
- 已实现文档元数据增强与过期过滤：
  - 文档 `metadata` 增加来源和更新时间。
  - 支持按 `stale-days` 过滤过期文档（时间敏感数据可失效）。
- 代码落点：
  - `model/RAG/retrieve_utils.py`
  - `model/RAG/retrieve_model.py`
  - `config/config-web.yaml` -> `model.rag.indexing.stale-days`

### 2）文本分割：递归分割参数化
- 已实现递归分割参数配置化（chunk size / overlap）。
- 代码落点：
  - `model/RAG/retrieve_model.py`
  - `config/config-web.yaml` -> `model.rag.indexing.chunk-size` / `chunk-overlap`

### 3）向量化：支持 ZhipuAI embedding-3
- 已实现双引擎可切换：
  - `modelscope`（默认）
  - `zhipuai`（模型默认 `embedding-3`）
- API Key 支持单独配置（可回落复用 `LLM_API_KEY`）。
- 代码落点：
  - `model/RAG/retrieve_model.py`
  - `model/Internet/Internet_model.py`
  - `config/config-web.yaml` -> `model.embedding.*`
  - `.env.example` -> `EMBEDDING_API_KEY`

### 4）数据入库：Neo4j 图 + 向量检索能力
- 已实现 Neo4j 向量检索桥接（可选 provider）：
  - 支持把分块写入 Neo4j 并创建向量索引检索。
  - 默认仍保留 FAISS，本地可平滑回退。
- 已实现 KG 关系类型过滤与限流（降噪）：
  - 只保留白名单关系类型，控制关系条数上限。
- 代码落点：
  - `model/RAG/neo4j_vector_bridge.py`
  - `model/RAG/retrieve_model.py` -> `model.rag.vector-store.*`
  - `qa/kg_relation_filter.py`
  - `qa/function_tool.py`

### 5）索引工程：多表示索引 + 多模态摘要 + Parent Retrieval
- 已实现多模态摘要索引基础能力：
  - 图像/表格生成可检索摘要文档。
  - 保留 `source_doc_id` 与父文档映射，召回后回源完整内容。
- 已实现 RAPTOR-lite 摘要层索引（分层召回）。
- 代码落点：
  - `model/RAG/multimodal_index.py`
  - `model/RAG/raptor_lite.py`
  - `model/RAG/retrieve_model.py`
  - `model/RAG/retrieve_service.py`

## 二、在线应用阶段

### 1）用户提问环节：问题改写 + Follow-up 压缩 + Multi-Query
- 已实现：
  - 历史对话相关问题改写为独立问题（follow-up 压缩）。
  - 多查询生成与融合去重。
  - 检索策略自动选择（单问/多问）。
- 代码落点：
  - `rag/query_optimizer.py`
  - `rag/retrieval_fusion.py`
  - `rag/retrieval_strategy.py`
  - `rag/retrieve/retrieve_document.py`
  - `rag/rag_chain.py`

### 2）数据检索环节：RAPTOR-lite + 策略增强
- 已实现：
  - 摘要层优先召回、父文档回源。
  - 与多查询融合策略协同。
- 代码落点：
  - `model/RAG/raptor_lite.py`
  - `model/RAG/retrieve_service.py`
  - `rag/rag_chain.py`

### 3）生成环节优化：Self-RAG 轻闭环
- 已实现：
  - 初稿生成 -> 自检评分 -> 低分重检索 -> 最终回答。
  - 失败回退与超时/重试上限保障可用性。
- 代码落点：
  - `rag/self_rag.py`
  - `rag/rag_chain.py`
  - `config/config-web.yaml` -> `model.rag.self-rag.*`

### 4）评估阶段：RAGAS 兼容 + 运行期指标
- 已实现：
  - 基础指标、场景指标、失败原因分布。
  - `ragas + datasets` 安装时走真实指标；未安装时优雅降级。
  - 运行期可观测指标（planner/memory/self-rag）。
- 代码落点：
  - `evaluation/rag_eval_runner.py`
  - `evaluation/sample_dataset.jsonl`
  - `utils/observability.py`

## 三、Agent 能力阶段

### 1）Planning（ReAct 思路）
- 已实现轻量规划器：
  - 在 `text / RAG / KnowledgeGraph / InternetSearch` 间路由。
  - 默认不覆盖显式意图（可配置）。
- 代码落点：
  - `qa/agent_planner.py`
  - `qa/agent_orchestrator.py`
  - `qa/answer.py`

### 2）Memory（短期 + 长期，显性 + 隐性）
- 已实现：
  - 短期记忆压缩（保留最近轮次、控制消息长度）。
  - 长期记忆抽取与召回（疾病史/过敏史/偏好等）。
  - 持久化向量长期记忆（JSONL + 向量检索，支持重启后恢复）。
- 代码落点：
  - `qa/agent_memory.py`
  - `qa/vector_memory_store.py`
  - `qa/agent_orchestrator.py`
  - `config/config-web.yaml` -> `model.agent.memory.*`

### 3）Tools / Action（MCP + A2A）
- MCP：
  - 已实现外部 MCP 目录加载、问题相关 MCP 推荐、可选建议注入。
- A2A：
  - 已实现 HTTP/JSON-RPC 适配层。
  - 已实现主链可配置委派（assist/direct 模式），失败不打断主流程。
- 代码落点：
  - `qa/external_ecosystem.py`
  - `qa/a2a_adapter.py`
  - `qa/agent_orchestrator.py`
  - `docs/integration/2026-03-03-mcp-a2a-integration-guide.md`

## 四、测试与验收结果

### 1）全量单元测试
- 命令：
  - `KMP_DUPLICATE_LIB_OK=TRUE .venv/Scripts/python -m unittest discover -s tests -p "test_*.py" -v`
- 结果：
  - `Ran 75 tests ... OK`

### 2）语法编译检查
- 命令：
  - `.venv/Scripts/python -m compileall -q app.py qa rag model Internet evaluation utils config env.py`
- 结果：
  - 通过（无语法错误）

### 3）真实链路烟测（含 RAG/KG/联网）
- 命令：
  - `.venv/Scripts/python evaluation/rag_eval_runner.py --dataset evaluation/sample_dataset.jsonl --output evaluation/reports/latest_report.json`
- 结果：
  - `total=3, answered=3, errors=0`
  - 场景命中：RAG / KnowledgeGraph / InternetSearch 均返回有效结果

## 五、结论

按 `Polish.md` 的离线→在线→Agent 思考顺序，需求已完成并通过全量测试与真实烟测。当前版本可直接进入手动网页回归与下一轮性能/效果精修。
