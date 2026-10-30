from app.config import Settings


def get_chat_model(settings: Settings):
  from langchain_openai import ChatOpenAI

  return ChatOpenAI(
      model=settings.chat_model,
      temperature=0,
      api_key=settings.openai_api_key or None,
      base_url=settings.openai_base_url,
  )


def get_embeddings(settings: Settings):
  from langchain_openai import OpenAIEmbeddings

  kwargs = {}
  if settings.embedding_model.startswith("text-embedding-3"):
    kwargs["dimensions"] = settings.embedding_dim  # lets -3-large be shortened to fit pgvector's HNSW limit
  return OpenAIEmbeddings(
      model=settings.embedding_model,
      api_key=settings.openai_api_key or None,
      base_url=settings.openai_base_url,
      **kwargs,
  )
