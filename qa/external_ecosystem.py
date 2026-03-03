from dataclasses import dataclass
from typing import Iterable, List

from qa.purpose_type import userPurposeType


@dataclass
class ExternalMcpServer:
    id: str
    name: str
    description: str
    tags: List[str]
    endpoint: str = ""
    repository: str = ""
    source: str = ""


DEFAULT_MCP_CATALOG = [
    ExternalMcpServer(
        id="ai.exa/exa",
        name="Exa MCP",
        description="Fast web search and crawling for retrieval.",
        tags=["search", "retrieval", "web", "research"],
        endpoint="https://mcp.exa.ai/mcp",
        repository="https://github.com/exa-labs/exa-mcp-server",
        source="mcp-registry",
    ),
    ExternalMcpServer(
        id="ai.com.mcp/hapi-mcp",
        name="HAPI MCP",
        description="Expose OpenAPI REST APIs as MCP tools.",
        tags=["structured", "api", "openapi", "medical", "integration"],
        endpoint="https://{HAPI_FQDN}:{HAPI_PORT}/mcp",
        repository="https://github.com/larebelion/hapimcp",
        source="mcp-registry",
    ),
    ExternalMcpServer(
        id="ai.com.mcp/registry",
        name="MCP Registry MCP",
        description="Discover and publish MCP servers.",
        tags=["registry", "discovery", "ecosystem"],
        endpoint="https://registry.modelcontextprotocol.io/v0.1/servers",
        repository="https://github.com/modelcontextprotocol/registry",
        source="mcp-registry",
    ),
    ExternalMcpServer(
        id="ai.com.mcp/skills-search",
        name="Agent Skills Search MCP",
        description="Search skills from skills.sh registry.",
        tags=["skills", "discovery", "search"],
        endpoint="https://skills-sh.run.mcp.com.ai/mcp",
        repository="https://github.com/agentskills/agentskills",
        source="mcp-registry",
    ),
]


_MEDICAL_KEYWORDS = (
    "医疗",
    "医学",
    "疾病",
    "药",
    "临床",
    "症状",
    "治疗",
    "副作用",
    "试验",
)
_SEARCH_KEYWORDS = ("搜索", "检索", "查找", "资料", "最新", "文献", "综述")
_STRUCTURED_KEYWORDS = ("结构化", "接口", "api", "openapi", "标准")


def _norm_tags(tags: Iterable[str]) -> List[str]:
    result = []
    for tag in tags or []:
        text = str(tag).strip().lower()
        if text:
            result.append(text)
    return result


def load_mcp_catalog(raw_catalog) -> List[ExternalMcpServer]:
    if not isinstance(raw_catalog, list):
        return []

    servers: List[ExternalMcpServer] = []
    for row in raw_catalog:
        if not isinstance(row, dict):
            continue
        server_id = str(row.get("id", "")).strip()
        name = str(row.get("name", "")).strip()
        if not server_id or not name:
            continue
        servers.append(
            ExternalMcpServer(
                id=server_id,
                name=name,
                description=str(row.get("description", "")).strip(),
                tags=_norm_tags(row.get("tags", [])),
                endpoint=str(row.get("endpoint", "")).strip(),
                repository=str(row.get("repository", "")).strip(),
                source=str(row.get("source", "")).strip(),
            )
        )
    return servers


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(word in text for word in keywords)


def _server_score(
    question: str,
    purpose: userPurposeType,
    server: ExternalMcpServer,
) -> int:
    q = (question or "").lower()
    tags = set(_norm_tags(server.tags))
    score = 0

    if purpose in (userPurposeType.RAG, userPurposeType.InternetSearch):
        if {"search", "retrieval", "web", "research"} & tags:
            score += 2
    if purpose == userPurposeType.KnowledgeGraph and {"structured", "api"} & tags:
        score += 1

    if _contains_any(q, _SEARCH_KEYWORDS) and {"search", "retrieval", "web"} & tags:
        score += 3
    if _contains_any(q, _MEDICAL_KEYWORDS) and {"medical", "clinical", "drug", "api"} & tags:
        score += 3
    if _contains_any(q, _STRUCTURED_KEYWORDS) and {"structured", "api", "openapi"} & tags:
        score += 2

    if "registry" in tags or "discovery" in tags:
        score += 1
    return score


def recommend_mcp_servers(
    question: str,
    purpose: userPurposeType,
    catalog: List[ExternalMcpServer],
    max_items: int = 3,
) -> List[ExternalMcpServer]:
    if max_items <= 0:
        return []
    if not catalog:
        return []

    scored = []
    for server in catalog:
        score = _server_score(question, purpose, server)
        if score > 0:
            scored.append((score, server.name.lower(), server))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [row[2] for row in scored[:max_items]]
    return selected


def build_external_tool_advice(
    question: str,
    purpose: userPurposeType,
    catalog: List[ExternalMcpServer],
    max_items: int = 3,
    a2a_enabled: bool = False,
) -> str:
    picked = recommend_mcp_servers(
        question=question,
        purpose=purpose,
        catalog=catalog,
        max_items=max_items,
    )
    if not picked and not a2a_enabled:
        return ""

    lines = ["外部工具建议（可选）："]
    for server in picked:
        endpoint = f" | endpoint: {server.endpoint}" if server.endpoint else ""
        lines.append(f"- {server.name} [{server.id}]：{server.description}{endpoint}")

    if a2a_enabled:
        lines.append("- A2A 已启用：可将复杂子任务委派给外部 Agent。")
    return "\n".join(lines)
