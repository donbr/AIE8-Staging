# %%
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from langchain_core.documents import Document

class State(TypedDict):
  question: str
  context: list[Document]
  response: str

# %%
import nest_asyncio

nest_asyncio.apply()

# %%
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader

directory_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)

ai_usage_knowledge_resources = directory_loader.load()

# %%
ai_usage_knowledge_resources[0].page_content[:1000]

# %%
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

def tiktoken_len(text):
    # Using cl100k_base encoding which is a good general-purpose tokenizer
    # This works well for estimating token counts even with Ollama models
    tokens = tiktoken.get_encoding("cl100k_base").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 750,
    chunk_overlap = 0,
    length_function = tiktoken_len,
)

# %%
ai_usage_knowledge_chunks = text_splitter.split_documents(ai_usage_knowledge_resources)

# %%
from langchain_ollama import OllamaEmbeddings
 
# Using embeddinggemma which is a powerful open-source embedding model
embedding_model = OllamaEmbeddings(model="embeddinggemma:latest")

# %%
embedding_dim = 768

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

# %%
client.create_collection(
    collection_name="ai_usage_knowledge_index",
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)

# %%
vector_store = QdrantVectorStore(
    client=client,
    collection_name="ai_usage_knowledge_index",
    embedding=embedding_model,
)

# %%
_ = vector_store.add_documents(documents=ai_usage_knowledge_chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retriever.invoke("How do people use AI in their daily work?")

# %%
def retrieve(state: State) -> State:
  retrieved_docs = retriever.invoke(state["question"])
  return {"context" : retrieved_docs}

# %%
from langchain_core.prompts import ChatPromptTemplate

HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provide context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context response with "I don't know"
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("human", HUMAN_TEMPLATE)
])

# %%
chat_prompt.invoke({"context" : "OUR CONTEXT HERE", "query" : "OUR QUERY HERE"}).messages[0].content

# %%
from langchain_ollama import ChatOllama

# Using gpt-oss:20b which is a powerful and efficient local model
ollama_chat_model = ChatOllama(model="gpt-oss:20b", temperature=0.6)

# %%
ollama_chat_model.invoke(chat_prompt.invoke({"context" : "Paris is the capital of France", "query" : "What is the capital of France?"}))

# %%
from langchain_core.output_parsers import StrOutputParser

generator_chain = chat_prompt | ollama_chat_model | StrOutputParser()

generator_chain.invoke({"context" : "Paris is the capital of France", "query" : "What is the capital of France?"})

# %%
def generate(state: State) -> State:
  generator_chain = chat_prompt | ollama_chat_model | StrOutputParser()
  response = generator_chain.invoke({"query" : state["question"], "context" : state["context"]})
  return {"response" : response}

# %%
graph_builder = StateGraph(State)
graph_builder = graph_builder.add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# %%
graph

# %%
from IPython.display import Markdown, display
response = graph.invoke({"question" : "What are the most common ways people use AI in their work?"})
display(Markdown(response["response"]))

# %%
response = graph.invoke({"question" : "Do people use AI for their personal lives?"})
display(Markdown(response["response"]))

# %%
response = graph.invoke({"question" : "What concerns or challenges do people have when using AI?"})
display(Markdown(response["response"]))

# %%
response = graph.invoke({"question" : "Who is Batman?"})
display(Markdown(response["response"]))

# %%



