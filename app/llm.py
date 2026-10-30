from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from app.config import Settings


def get_chat_model(settings: "Settings"):
  from langchain_openai import ChatOpenAI

  return ChatOpenAI(
      model=settings.chat_model,
      temperature=0,
      api_key=settings.openai_api_key or None,
      base_url=settings.openai_base_url,
      timeout=settings.llm_timeout,
  )


def get_embeddings(settings: "Settings"):
  from langchain_openai import OpenAIEmbeddings

  kwargs: dict = {}
  if settings.embedding_model.startswith("text-embedding-3"):
    kwargs["dimensions"] = settings.embedding_dim  # lets -3-large be shortened to fit pgvector's HNSW limit
  if settings.openai_base_url:
    # Non-OpenAI servers (Ollama, gateways) expect plain strings. By default the client
    # pre-tokenizes text with OpenAI's tokenizer and sends token ids, which they reject.
    kwargs["check_embedding_ctx_length"] = False
  return OpenAIEmbeddings(
      model=settings.embedding_model,
      api_key=settings.openai_api_key or None,
      base_url=settings.openai_base_url,
      timeout=settings.llm_timeout,
      **kwargs,
  )


def structured(llm: Any, schema: Any, method: str = ""):
  """llm.with_structured_output(schema), optionally forcing how JSON output is requested."""
  if method:
    return llm.with_structured_output(schema, method=method)
  return llm.with_structured_output(schema)
