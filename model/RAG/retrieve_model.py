"""本地知识库的 RAG 检索模型类。"""

import os
import shutil
import hashlib
from typing import List

from config.config import Config
from env import get_env_value
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    MHTMLLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modelscope.hub.snapshot_download import snapshot_download
from model.RAG.retrieve_utils import attach_last_modified_metadata, filter_stale_documents
from model.RAG.multimodal_index import build_multimodal_documents
from model.RAG.neo4j_vector_bridge import build_neo4j_vector_retriever
from model.RAG.raptor_lite import build_summary_layer
from model.model_base import ModelStatus, Modelbase
from py2neo import Graph


class Retrievemodel(Modelbase):
    _retriever: object | None
    _summary_retriever: object | None

    def __init__(self, *args, **krgs):
        super().__init__(*args, **krgs)
        self._retriever = None
        self._summary_retriever = None
        self._user_retrievers = {}
        self._source_chunks_map = {}
        self._multimodal_parent_map = {}

        self._embedding_provider = str(
            self._get_config_value(["model", "embedding", "provider"], "modelscope")
        ).lower()
        self._embedding_download_path = self._get_config_value(
            ["model", "embedding", "model-path"], os.path.expanduser("~/.cache/modelscope/hub")
        )
        self._embedding_model_name = self._get_config_value(
            ["model", "embedding", "model-name"], "iic/nlp_corom_sentence-embedding_chinese-base"
        )
        self._embedding_model_path = os.path.join(
            self._embedding_download_path, self._embedding_model_name
        )
        self._embedding_api_key_env = self._get_config_value(
            ["model", "embedding", "api-key-env"], "EMBEDDING_API_KEY"
        )
        self._embedding_zhipu_model = self._get_config_value(
            ["model", "embedding", "zhipu-model"], "embedding-3"
        )

        self._chunk_size = self._to_int(
            self._get_config_value(["model", "rag", "indexing", "chunk-size"], 2000), 2000
        )
        self._chunk_overlap = self._to_int(
            self._get_config_value(["model", "rag", "indexing", "chunk-overlap"], 100), 100
        )
        self._stale_days = self._to_int(
            self._get_config_value(["model", "rag", "indexing", "stale-days"], 0), 0
        )
        self._retriever_k = self._to_int(
            self._get_config_value(["model", "rag", "retrieval", "top-k-per-query"], 6), 6
        )
        self._vector_store_provider = str(
            self._get_config_value(["model", "rag", "vector-store", "provider"], "faiss")
        ).strip().lower()
        self._neo4j_vector_label = str(
            self._get_config_value(["model", "rag", "vector-store", "neo4j", "label"], "RagChunk")
        ).strip()
        self._neo4j_vector_index = str(
            self._get_config_value(
                ["model", "rag", "vector-store", "neo4j", "index-name"], "rag_chunks_index"
            )
        ).strip()
        self._neo4j_text_property = str(
            self._get_config_value(
                ["model", "rag", "vector-store", "neo4j", "text-property"], "text"
            )
        ).strip()
        self._neo4j_embedding_property = str(
            self._get_config_value(
                ["model", "rag", "vector-store", "neo4j", "embedding-property"], "embedding"
            )
        ).strip()
        self._neo4j_metadata_property = str(
            self._get_config_value(
                ["model", "rag", "vector-store", "neo4j", "metadata-property"], "metadata_json"
            )
        ).strip()
        self._neo4j_reset_on_build = self._to_bool(
            self._get_config_value(
                ["model", "rag", "vector-store", "neo4j", "reset-on-build"], True
            ),
            True,
        )
        self._raptor_enabled = self._to_bool(
            self._get_config_value(["model", "rag", "raptor-lite", "enabled"], False),
            False,
        )
        self._raptor_summary_max_chars = self._to_int(
            self._get_config_value(
                ["model", "rag", "raptor-lite", "summary-max-chars"],
                260,
            ),
            260,
        )
        self._raptor_summary_search_k = self._to_int(
            self._get_config_value(["model", "rag", "raptor-lite", "summary-search-k"], 6),
            6,
        )

        self._embedding = self._build_embedding_model()

        self._data_path = self._get_config_value(["Knowledge-base-path"], "./konwledge-base")
        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)

    def _get_config_value(self, path: List[str], default):
        try:
            return Config.get_instance().get_with_nested_params(*path)
        except Exception:
            return default

    @staticmethod
    def _to_int(value, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _to_bool(value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("1", "true", "yes", "on"):
                return True
            if text in ("0", "false", "no", "off"):
                return False
        return default

    def _ensure_modelscope_model(self):
        if os.path.exists(self._embedding_model_path):
            return
        try:
            model_dir = snapshot_download(
                self._embedding_model_name,
                cache_dir=self._embedding_download_path,
            )
            print(f"Model downloaded and saved to {model_dir}")
        except Exception as e:
            print(f"Failed to download model: {e}")
            if os.path.exists(self._embedding_model_path):
                shutil.rmtree(self._embedding_model_path)
            raise

    def _build_modelscope_embedding(self):
        self._ensure_modelscope_model()
        return ModelScopeEmbeddings(model_id=self._embedding_model_path)

    def _build_embedding_model(self):
        if self._embedding_provider == "zhipuai":
            api_key = get_env_value(self._embedding_api_key_env) or get_env_value("LLM_API_KEY")
            if api_key:
                try:
                    from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings

                    return ZhipuAIEmbeddings(
                        api_key=api_key,
                        model=self._embedding_zhipu_model,
                    )
                except Exception as e:
                    print(f"Zhipu embedding unavailable, fallback to modelscope: {e}")
            else:
                print(
                    f"Missing embedding api key env '{self._embedding_api_key_env}', fallback to modelscope"
                )

        return self._build_modelscope_embedding()

    def _get_neo4j_graph(self):
        try:
            url = self._get_config_value(["database", "neo4j", "url"], "")
            username = self._get_config_value(["database", "neo4j", "username"], "")
            password = self._get_config_value(["database", "neo4j", "password"], "")
            if not url:
                return None
            return Graph(url, auth=(username, password))
        except Exception as e:
            print(f"连接 Neo4j 向量库失败，将回退 FAISS: {e}")
            return None

    def _load_documents_from_path(self, base_path: str) -> List[Document]:
        pdf_docs = DirectoryLoader(
            base_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
            use_multithreading=True,
        ).load()
        docx_docs = DirectoryLoader(
            base_path,
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
            silent_errors=True,
            use_multithreading=True,
        ).load()
        txt_docs = DirectoryLoader(
            base_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            silent_errors=True,
            loader_kwargs={"autodetect_encoding": True},
            use_multithreading=True,
        ).load()
        csv_docs = DirectoryLoader(
            base_path,
            glob="**/*.csv",
            loader_cls=CSVLoader,
            silent_errors=True,
            loader_kwargs={"autodetect_encoding": True},
            use_multithreading=True,
        ).load()
        html_docs = DirectoryLoader(
            base_path,
            glob="**/*.html",
            loader_cls=UnstructuredHTMLLoader,
            silent_errors=True,
            use_multithreading=True,
        ).load()
        mhtml_docs = DirectoryLoader(
            base_path,
            glob="**/*.mhtml",
            loader_cls=MHTMLLoader,
            silent_errors=True,
            use_multithreading=True,
        ).load()
        markdown_docs = DirectoryLoader(
            base_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            silent_errors=True,
            use_multithreading=True,
        ).load()

        docs = (
            pdf_docs
            + docx_docs
            + txt_docs
            + csv_docs
            + html_docs
            + mhtml_docs
            + markdown_docs
        )
        multimodal_docs, multimodal_parent_map = build_multimodal_documents(
            base_path,
            max_summary_chars=self._raptor_summary_max_chars,
        )
        docs = docs + multimodal_docs
        self._multimodal_parent_map = multimodal_parent_map
        docs = attach_last_modified_metadata(docs)
        docs = filter_stale_documents(docs, stale_days=self._stale_days)
        return docs

    def _build_retriever_from_docs(
        self,
        docs: List[Document],
        enable_summary_index: bool = True,
    ) -> object | None:
        if not docs:
            return None
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        splits = splitter.split_documents(docs)
        if not splits:
            return None

        retriever = None
        if self._vector_store_provider == "neo4j":
            graph = self._get_neo4j_graph()
            if graph is not None:
                try:
                    retriever = build_neo4j_vector_retriever(
                        graph=graph,
                        embedding_model=self._embedding,
                        docs=splits,
                        index_name=self._neo4j_vector_index,
                        label=self._neo4j_vector_label,
                        top_k=self._retriever_k,
                        embedding_property=self._neo4j_embedding_property,
                        text_property=self._neo4j_text_property,
                        metadata_property=self._neo4j_metadata_property,
                        reset_before_build=self._neo4j_reset_on_build,
                    )
                except Exception as e:
                    print(f"构建 Neo4j 向量检索失败，回退 FAISS: {e}")

        if retriever is None:
            vectorstore = FAISS.from_documents(documents=splits, embedding=self._embedding)
            retriever = vectorstore.as_retriever(search_kwargs={"k": self._retriever_k})

        if enable_summary_index:
            self._summary_retriever = None
            self._source_chunks_map = {}

        if self._raptor_enabled and enable_summary_index:
            summary_docs, source_map = build_summary_layer(
                splits, max_summary_chars=self._raptor_summary_max_chars
            )
            if summary_docs:
                try:
                    summary_store = FAISS.from_documents(
                        documents=summary_docs,
                        embedding=self._embedding,
                    )
                    self._summary_retriever = summary_store.as_retriever(
                        search_kwargs={"k": self._raptor_summary_search_k}
                    )
                    merged_map = dict(source_map)
                    merged_map.update(self._multimodal_parent_map)
                    self._source_chunks_map = merged_map
                except Exception as e:
                    self._summary_retriever = None
                    self._source_chunks_map = {}
                    print(f"构建摘要索引失败，已回退普通检索: {e}")
        elif enable_summary_index and self._multimodal_parent_map:
            self._source_chunks_map = dict(self._multimodal_parent_map)

        return retriever

    # 建立向量库
    def build(self):
        self._model_status = ModelStatus.BUILDING
        try:
            docs = self._load_documents_from_path(self._data_path)
            self._retriever = self._build_retriever_from_docs(
                docs, enable_summary_index=True
            )
            self._model_status = (
                ModelStatus.READY if self._retriever is not None else ModelStatus.INVALID
            )
        except Exception as e:
            self._model_status = ModelStatus.FAILED
            print(f"构建知识库向量库失败: {e}")

    @property
    def retriever(self):
        if self._retriever is None or self._model_status in (
            ModelStatus.FAILED,
            ModelStatus.INVALID,
        ):
            self.build()

        if self._retriever is None:
            raise RuntimeError("知识库向量库不可用，请检查知识库文件和配置")

        return self._retriever

    def build_user_vector_store(self):
        """根据用户 ID 加载用户文件夹中的文件并构建向量库。"""
        user_data_path = os.path.join("user_data", self.user_id)
        if not os.path.exists(user_data_path):
            print(f"用户文件夹 {user_data_path} 不存在")
            return

        try:
            if self.user_id in self._user_retrievers:
                del self._user_retrievers[self.user_id]
                print(f"用户 {self.user_id} 的旧向量库已删除")

            docs = self._load_documents_from_path(user_data_path)
            if not docs:
                print(f"用户 {self.user_id} 文件夹中没有找到文档")
                return

            retriever = self._build_retriever_from_docs(
                docs,
                enable_summary_index=False,
            )
            if retriever is None:
                print(f"用户 {self.user_id} 的文档无法构建有效向量库")
                return

            self._user_retrievers[self.user_id] = retriever
            print(f"用户 {self.user_id} 的向量库已构建完成")
        except Exception as e:
            print(f"构建用户 {self.user_id} 向量库时出错: {e}")

    def get_user_retriever(self):
        """获取用户 retriever，如果不存在返回 None。"""
        return self._user_retrievers.get(self.user_id, None)

    @property
    def summary_retriever(self):
        return self._summary_retriever

    def retrieve_chunks_by_source_ids(
        self, source_ids: List[str], limit: int = 8
    ) -> List[Document]:
        if not source_ids:
            return []

        docs: List[Document] = []
        seen = set()
        for source_id in source_ids:
            for doc in self._source_chunks_map.get(source_id, []):
                source = ""
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    source = str(doc.metadata.get("source", ""))
                content = str(getattr(doc, "page_content", ""))
                key = hashlib.sha1(f"{source}\n{content}".encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
                if limit > 0 and len(docs) >= limit:
                    return docs
        return docs

    def upload_user_file(self, file):
        """将用户上传的文件存储到用户文件夹。"""
        user_data_path = os.path.join("user_data", self.user_id)
        os.makedirs(user_data_path, exist_ok=True)

        file_path = os.path.join(user_data_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        print(f"文件 {file.name} 已成功上传到用户 {self.user_id} 的文件夹")

    def list_uploaded_files(self):
        """展示用户已上传文件。"""
        user_data_path = os.path.join("user_data", self.user_id)
        if not os.path.exists(user_data_path):
            print(f"用户文件夹 {user_data_path} 不存在")
            return []

        files = os.listdir(user_data_path)
        if files:
            print(f"用户 {self.user_id} 已上传的文件：")
            for file in files:
                print(file)
        else:
            print(f"用户 {self.user_id} 文件夹为空")
        return files

    def delete_uploaded_file(self, filename=None):
        """删除指定文件或清空用户文件夹。"""
        user_data_path = os.path.join("user_data", self.user_id)
        if not os.path.exists(user_data_path):
            print(f"用户文件夹 {user_data_path} 不存在")
            return

        if filename:
            file_path = os.path.join(user_data_path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"文件 {filename} 已成功删除")
            else:
                print(f"文件 {filename} 不存在")
            return

        for file in os.listdir(user_data_path):
            file_path = os.path.join(user_data_path, file)
            os.remove(file_path)
        print(f"用户 {self.user_id} 文件夹已清空")

    def view_uploaded_file(self, filename):
        """根据文件名返回用户文件路径。"""
        user_data_path = os.path.join("user_data", self.user_id)
        file_path = os.path.join(user_data_path, filename)
        if not os.path.exists(file_path):
            print(f"文件 {filename} 不存在")
            return None
        print(f"文件 {filename} 路径已成功获取")
        return file_path


INSTANCE = Retrievemodel()
