# Cyber-Doctor 二期 Agent 优化方案

## 目标
在不破坏现有 Gradio 交互和一期检索链路的前提下，分批落地 Agent 能力：`Planning + Memory + Tool Action`，并预留 MCP/A2A 扩展接口。

## 二期总方案（完整）

### P2-Batch1：Planner + Memory 基础能力（本次已落地）
- 新增 ReAct 风格一步规划器：将模糊文本问题在 `text / RAG / KnowledgeGraph / InternetSearch` 间路由。
- 新增短期记忆压缩：对历史轮次和单条消息做裁剪，降低提示词噪音和上下文长度。
- 新增轻量长期记忆：抽取用户事实（姓名、年龄、过敏史、疾病史），并按会话召回注入问题。
- 在 `qa.answer` 主入口接入编排层，不影响图片/音频/PPT/Docx/Video 流程。

### P2-Batch2：检索策略升级（本次已落地）
- 基于一期多查询融合继续增强：
  - 会话相关问题触发“问题压缩 + 多查询”组合策略。
  - 增加检索策略选择器（短问答 vs 多约束问答）。
- 引入 RAPTOR-lite：
  - 离线生成“段落摘要层”索引（不做全树工程化）。
  - 在线优先召回摘要层，再回源完整文档。

### P2-Batch3：生成阶段反思（本次已落地）
- 在 `rag/rag_chain.py` 从当前 self-check 升级为轻量 Self-RAG 闭环：
  - 生成 -> 自检打分 -> 低分触发重检索 -> 最终回答。
- 加入最大重试/超时阈值，保证可用性优先。

### P2-Batch4：评估与可观测（本次已落地）
- 扩展 `evaluation/rag_eval_runner.py`：
  - 增加分场景指标（KG、RAG、Internet）。
  - 输出失败原因分布（检索空、上下文冲突、模型幻觉）。
- 增加运行期埋点：planner 动作分布、memory 命中率、重检索触发率。

### P2-Batch5：外部工具生态（本次已落地）
- MCP：梳理可直接接入的医疗/检索/结构化处理型 MCP Server。
- A2A：预留多 Agent 协作接口（仅定义协议适配层，不在当前轮做完整编排）。

## 当前已完成实现（Batch1 + Batch2 + Batch3 + Batch4 + Batch5）
- `qa/agent_planner.py`
- `qa/agent_memory.py`
- `qa/agent_orchestrator.py`
- `qa/answer.py` 接入编排
- `rag/retrieval_strategy.py`
- `model/RAG/raptor_lite.py`
- `model/RAG/retrieve_service.py` 增加 `retrieve_with_raptor_lite`
- `model/RAG/retrieve_model.py` 增加摘要层索引与回源映射
- `rag/rag_chain.py` 接入策略选择与 summary-first 检索
- `rag/self_rag.py` 轻量 Self-RAG 评估解析与重检索查询生成
- `rag/rag_chain.py` 接入低置信度重检索闭环（带重试与超时上限）
- `utils/observability.py` 运行期埋点统计（planner/memory/self-rag）
- `evaluation/rag_eval_runner.py` 增加场景指标、失败原因分布、运行期埋点输出
- `qa/external_ecosystem.py` MCP 目录加载、语义推荐与建议注入
- `qa/a2a_adapter.py` A2A HTTP/JSON-RPC 协议适配层（预留委派接口）
- `qa/agent_orchestrator.py` 增加 external tool-advice 可配置注入
- `config/config-web.yaml` 增加 `model.agent.*`
- 新增测试：
  - `tests/test_agent_planner.py`
  - `tests/test_agent_memory.py`
  - `tests/test_answer_agent_orchestration.py`
  - `tests/test_retrieval_strategy.py`
  - `tests/test_raptor_lite.py`
  - `tests/test_retrieve_service.py`
  - `tests/test_self_rag.py`
  - `tests/test_rag_chain_self_rag.py`
  - `tests/test_observability.py`
  - `tests/test_external_ecosystem.py`
  - `tests/test_a2a_adapter.py`

## 验收标准
- 文本类问题可自动进行 planner 路由与 memory 注入。
- 显式意图（根据知识库/知识图谱/搜索）默认不被 planner 覆盖。
- 历史对话在配置阈值下被压缩，且不会破坏原有流式回答流程。
- 全量单元测试通过。

## 风险与约束
- Planner 依赖 LLM 输出结构化 JSON，需继续观察线上格式稳定性。
- 长期记忆当前为轻量会话内存储，后续可替换为向量库或数据库持久化。
- `parse_question` 与 planner 存在双路决策，后续可合并为单一路由器减少重复调用。
