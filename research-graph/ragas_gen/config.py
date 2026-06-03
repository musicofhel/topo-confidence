from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.testset.persona import Persona

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"

NEO4J_BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")


def get_llm():
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    return llm_factory(model=OPENROUTER_MODEL, client=client)


def get_embeddings():
    lc_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(embeddings=lc_emb)


PERSONAS = [
    Persona(name="ML researcher", role_description="Precise technical vocabulary, cites specific methods and metrics"),
    Persona(name="Practitioner", role_description="Asks 'how do I use X' style questions, focuses on practical application"),
    Persona(name="Reviewer", role_description="Asks 'what evidence supports/contradicts X', focuses on evaluation and comparison"),
]

SYNTHESIZER_WEIGHTS = {
    "single_hop_specific": 0.55,
    "multi_hop_specific": 0.30,
    "multi_hop_abstract": 0.15,
}

STYLE_DISTRIBUTION = {
    "perfect_grammar": 0.50,
    "web_search_like": 0.30,
    "misspelled": 0.10,
    "poor_grammar": 0.10,
}

DEDUP_THRESHOLD = 0.85
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
SYNTHETIC_OUTPUT = EVAL_DIR / "synthetic-v1.json"
