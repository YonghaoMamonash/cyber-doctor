# 2-5 外部工具生态落地说明（MCP + A2A）

日期：2026-03-03

## 本批目标
- MCP：提供可落地的外部工具目录、问题相关性推荐、可配置提示注入。
- A2A：提供协议适配层（HTTP + JSON-RPC），预留多 Agent 委派接口，不强制接入主链。

## 代码落点
- MCP 目录与推荐：`qa/external_ecosystem.py`
- A2A 适配器：`qa/a2a_adapter.py`
- 主链接入点（默认关闭）：`qa/agent_orchestrator.py`
- 配置：`config/config-web.yaml` 的 `model.agent.external.*`

## 默认行为
- 默认 `model.agent.external.tool-advice.enabled=false`，不会改变现有回答行为。
- 仅在开启后，才会在问题中附加“外部工具建议（可选）”段落。

## 推荐 MCP 候选（当前配置）
1. `ai.exa/exa`
- 用途：联网检索与网页抓取增强（适合 InternetSearch/RAG）。
- 源：MCP Registry（远端 `https://mcp.exa.ai/mcp`）。

2. `ai.com.mcp/hapi-mcp`
- 用途：将 OpenAPI 接口动态暴露为 MCP 工具。
- 对本项目价值：可以把医疗类 REST API（如药品、临床试验数据）包装成标准 MCP 能力。

3. `ai.com.mcp/registry`
- 用途：自动发现 MCP 服务器，便于后续扩展与版本管理。

4. `ai.com.mcp/skills-search`
- 用途：发现可复用 Agent Skills，支持能力拼装。

## 医疗方向落地建议
- 优先路线：`HAPI MCP + 医疗 OpenAPI`。
- 理由：Registry 中“医疗垂类 MCP”官方收录相对少，HAPI 路线更稳定、可控。
- 实施方式：把目标医疗 API（如药品说明、不良反应、临床试验）在网关侧统一为 OpenAPI，再通过 HAPI MCP 暴露给 Agent。

## A2A 适配层说明
- 适配器对外接口：`A2AHttpAdapter.send_text(...)`
- 当前采用 JSON-RPC 请求方法：`message/send`
- 请求结构：`jsonrpc/id/method/params(message/taskId/contextId/metadata)`
- 返回解析：优先读取 `result.artifacts[].parts[].text`，回退读取 `result.history[].parts[].text`
- 失败策略：网络错误/JSON-RPC error 均返回 `success=False`，不中断主链

## 启用示例
```yaml
model:
  agent:
    external:
      tool-advice:
        enabled: true
        max-mcp-suggestions: 2
      a2a:
        enabled: true
        endpoint: "http://127.0.0.1:8788/rpc"
        timeout-seconds: 15
```

## 参考来源
- MCP Server Concepts: https://modelcontextprotocol.io/docs/learn/server-concepts
- MCP Registry FAQ/API: https://registry.modelcontextprotocol.io/faq
- MCP Registry OpenAPI: https://registry.modelcontextprotocol.io/openapi.yaml
- Exa MCP Repo: https://github.com/exa-labs/exa-mcp-server
- HAPI MCP Repo: https://github.com/larebelion/hapimcp
- A2A Protocol 文档: https://a2aproject.github.io/A2A/latest/
