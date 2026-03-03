import unittest

from qa.external_ecosystem import (
    ExternalMcpServer,
    build_external_tool_advice,
    load_mcp_catalog,
    recommend_mcp_servers,
)
from qa.purpose_type import userPurposeType


class ExternalEcosystemTests(unittest.TestCase):
    def test_load_mcp_catalog(self):
        raw = [
            {
                "id": "exa",
                "name": "Exa Search",
                "description": "web search",
                "tags": ["search", "retrieval"],
            }
        ]
        catalog = load_mcp_catalog(raw)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0].id, "exa")

    def test_recommend_mcp_servers_for_medical_query(self):
        catalog = [
            ExternalMcpServer(
                id="openfda",
                name="OpenFDA via HAPI",
                description="drug labels and safety",
                tags=["medical", "drug", "structured", "api"],
            ),
            ExternalMcpServer(
                id="exa",
                name="Exa Search",
                description="web retrieval",
                tags=["search", "retrieval"],
            ),
        ]
        picked = recommend_mcp_servers(
            question="我想查药品不良反应和说明书",
            purpose=userPurposeType.InternetSearch,
            catalog=catalog,
            max_items=2,
        )
        self.assertEqual(picked[0].id, "openfda")

    def test_build_external_tool_advice(self):
        catalog = [
            ExternalMcpServer(
                id="exa",
                name="Exa Search",
                description="web retrieval",
                tags=["search", "retrieval"],
            ),
        ]
        advice = build_external_tool_advice(
            question="帮我检索最新医学综述",
            purpose=userPurposeType.RAG,
            catalog=catalog,
            max_items=1,
            a2a_enabled=True,
        )
        self.assertIn("外部工具建议", advice)
        self.assertIn("Exa Search", advice)
        self.assertIn("A2A", advice)


if __name__ == "__main__":
    unittest.main()
