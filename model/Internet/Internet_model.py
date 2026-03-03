"""联网搜索的 RAG 检索模型类。"""

import os
import shutil
from typing import List

from config.config import Config
from env import get_app_root, get_env_value
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.Internet.local_loader import load_local_html_documents
from model.RAG.retrieve_utils import attach_last_modified_metadata, filter_stale_documents
from model.model_base import ModelStatus, Modelbase
from modelscope.hub.snapshot_download import snapshot_download


class InternetModel(Modelbase):
    _retriever: VectorStoreRetriever | None

    def __init__(self, *args, **krgs):
        super().__init__(*args, **krgs)
        self._retriever = None

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

        self._embedding = self._build_embedding_model()
        self._data_path = os.path.join(get_app_root(), "data/cache/internet")

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

    def _ensure_modelscope_model(self):
        if os.path.exists(self._embedding_model_path):
            return
        try:
            snapshot_download(
                self._embedding_model_name,
                cache_dir=self._embedding_download_path,
            )
        except Exception as e:
            print(f"Failed to download modelscope embedding: {e}")
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

    def _load_documents(self) -> List[Document]:
        docs = load_local_html_documents(self._data_path)
        docs = attach_last_modified_metadata(docs)
        docs = filter_stale_documents(docs, stale_days=self._stale_days)
        return docs

    # 建立向量库
    def build(self):
        self._model_status = ModelStatus.BUILDING
        try:
            docs = self._load_documents()
            if not docs:
                self._retriever = None
                self._model_status = ModelStatus.INVALID
                return

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            splits = splitter.split_documents(docs)
            if not splits:
                self._retriever = None
                self._model_status = ModelStatus.INVALID
                return

            vectorstore = FAISS.from_documents(documents=splits, embedding=self._embedding)
            self._retriever = vectorstore.as_retriever(
                search_kwargs={"k": self._retriever_k}
            )
            self._model_status = ModelStatus.READY
        except Exception as e:
            self._retriever = None
            self._model_status = ModelStatus.FAILED
            print(f"构建联网检索向量库失败: {e}")

    @property
    def retriever(self) -> VectorStoreRetriever:
        if self._retriever is None or self._model_status in (
            ModelStatus.FAILED,
            ModelStatus.INVALID,
        ):
            self.build()
        if self._retriever is None:
            raise RuntimeError("联网检索向量库不可用")
        return self._retriever


INSTANCE = InternetModel()
