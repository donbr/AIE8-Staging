"""MCP Vector Store Server.

A Model Context Protocol server for vector database operations,
providing persistent document storage and similarity search capabilities
to replace in-memory Qdrant usage.

Features:
- Persistent vector storage with Qdrant
- Multiple collection management
- Advanced similarity search with metadata filtering
- Document chunking and embedding management
- Collection health monitoring and optimization
"""

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """A document chunk with metadata."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}
    collection: str = "default"
    timestamp: datetime = datetime.now()


class SearchResult(BaseModel):
    """Vector search result."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    collection: str


class CollectionStats(BaseModel):
    """Collection statistics."""
    name: str
    document_count: int
    vector_count: int
    index_size: int
    last_updated: datetime


class VectorStoreMCPServer:
    """MCP Server for vector database operations."""

    def __init__(self, qdrant_url: str = "http://localhost:6333", embedding_dim: int = 1536):
        self.qdrant_url = qdrant_url
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(url=qdrant_url)
        self.server = Server("vector-store")

        # Default collections
        self.default_collections = [
            "research_documents",
            "knowledge_base",
            "chat_history",
            "cached_embeddings"
        ]

        # Embedding service configuration
        self.embedding_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = "text-embedding-3-small"
        self.embedding_client = httpx.AsyncClient(timeout=30.0)

        # Cache for embeddings
        self.embedding_cache: Dict[str, List[float]] = {}

        # Register tools and resources
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        """Register MCP tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="vector_search",
                    description="Search for similar documents in the vector store",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query text"
                            },
                            "collection": {
                                "type": "string",
                                "description": "Collection name to search in",
                                "default": "research_documents"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "minimum": 1,
                                "maximum": 50,
                                "default": 5
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Minimum similarity score",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.7
                            },
                            "metadata_filter": {
                                "type": "object",
                                "description": "Metadata filters to apply",
                                "default": {}
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="add_documents",
                    description="Add documents to the vector store",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "documents": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "metadata": {"type": "object"},
                                        "id": {"type": "string"}
                                    },
                                    "required": ["content"]
                                },
                                "description": "List of documents to add"
                            },
                            "collection": {
                                "type": "string",
                                "description": "Collection name",
                                "default": "research_documents"
                            },
                            "chunk_size": {
                                "type": "integer",
                                "description": "Chunk size for splitting large documents",
                                "default": 750
                            },
                            "chunk_overlap": {
                                "type": "integer",
                                "description": "Overlap between chunks",
                                "default": 50
                            }
                        },
                        "required": ["documents"]
                    }
                ),
                Tool(
                    name="delete_documents",
                    description="Delete documents from the vector store",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "document_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of document IDs to delete"
                            },
                            "collection": {
                                "type": "string",
                                "description": "Collection name",
                                "default": "research_documents"
                            }
                        },
                        "required": ["document_ids"]
                    }
                ),
                Tool(
                    name="create_collection",
                    description="Create a new vector collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": "Name of the collection to create"
                            },
                            "vector_size": {
                                "type": "integer",
                                "description": "Dimension of vectors",
                                "default": 1536
                            },
                            "distance_metric": {
                                "type": "string",
                                "enum": ["cosine", "euclidean", "dot"],
                                "description": "Distance metric for similarity",
                                "default": "cosine"
                            }
                        },
                        "required": ["collection_name"]
                    }
                ),
                Tool(
                    name="list_collections",
                    description="List all available collections",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "vector_search":
                return await self._vector_search(**arguments)
            elif name == "add_documents":
                return await self._add_documents(**arguments)
            elif name == "delete_documents":
                return await self._delete_documents(**arguments)
            elif name == "create_collection":
                return await self._create_collection(**arguments)
            elif name == "list_collections":
                return await self._list_collections(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    def _register_resources(self):
        """Register MCP resources."""

        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="vector://collections/stats",
                    name="Collection Statistics",
                    description="Statistics for all vector collections",
                    mimeType="application/json"
                ),
                Resource(
                    uri="vector://health",
                    name="Vector Store Health",
                    description="Health status of the vector store",
                    mimeType="application/json"
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "vector://collections/stats":
                return await self._get_collection_stats()
            elif uri == "vector://health":
                return await self._get_health_status()
            else:
                raise ValueError(f"Unknown resource: {uri}")

    async def _vector_search(
        self,
        query: str,
        collection: str = "research_documents",
        limit: int = 5,
        threshold: float = 0.7,
        metadata_filter: Dict[str, Any] = {}
    ) -> List[TextContent]:
        """Perform vector similarity search."""
        try:
            # Ensure collection exists
            await self._ensure_collection_exists(collection)

            # Get query embedding
            query_embedding = await self._get_embedding(query)
            if not query_embedding:
                return [TextContent(
                    type="text",
                    text="Error: Could not generate embedding for query"
                )]

            # Build filter if provided
            search_filter = None
            if metadata_filter:
                search_filter = Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in metadata_filter.items()
                    ]
                )

            # Perform search
            search_result = self.client.search(
                collection_name=collection,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=threshold
            )

            # Format results
            if not search_result:
                return [TextContent(
                    type="text",
                    text=f"No results found for query: '{query}' in collection '{collection}'"
                )]

            results_text = [f"🔍 **Vector Search Results for: {query}**"]
            results_text.append(f"📊 Found {len(search_result)} results in '{collection}' collection")
            results_text.append("")

            for i, result in enumerate(search_result, 1):
                payload = result.payload or {}
                content = payload.get('content', 'No content')
                metadata = payload.get('metadata', {})

                results_text.extend([
                    f"**{i}. Document ID: {result.id}**",
                    f"📊 Score: {result.score:.4f}",
                    f"📝 Content: {content[:300]}{'...' if len(content) > 300 else ''}",
                ])

                if metadata:
                    results_text.append(f"🏷️ Metadata: {json.dumps(metadata, indent=2)}")

                results_text.append("")

            return [TextContent(
                type="text",
                text="\n".join(results_text)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Vector search error: {str(e)}"
            )]

    async def _add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection: str = "research_documents",
        chunk_size: int = 750,
        chunk_overlap: int = 50
    ) -> List[TextContent]:
        """Add documents to the vector store."""
        try:
            # Ensure collection exists
            await self._ensure_collection_exists(collection)

            added_count = 0
            points = []

            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                doc_id = doc.get('id', str(uuid.uuid4()))

                if not content:
                    continue

                # Chunk large documents
                chunks = self._chunk_text(content, chunk_size, chunk_overlap)

                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"

                    # Get embedding for chunk
                    embedding = await self._get_embedding(chunk)
                    if not embedding:
                        continue

                    # Add chunk metadata
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "parent_document_id": doc_id,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Create point
                    point = PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload={
                            "content": chunk,
                            "metadata": chunk_metadata
                        }
                    )
                    points.append(point)
                    added_count += 1

            # Batch insert points
            if points:
                self.client.upsert(
                    collection_name=collection,
                    points=points
                )

            return [TextContent(
                type="text",
                text=f"✅ Successfully added {added_count} document chunks to collection '{collection}'"
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error adding documents: {str(e)}"
            )]

    async def _delete_documents(
        self,
        document_ids: List[str],
        collection: str = "research_documents"
    ) -> List[TextContent]:
        """Delete documents from the vector store."""
        try:
            # Delete by document IDs (including chunks)
            deleted_count = 0

            for doc_id in document_ids:
                # Delete exact matches
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(
                        points=[doc_id]
                    )
                )

                # Delete chunks (document_id_chunk_*)
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(
                        filter=Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.parent_document_id",
                                    match=models.MatchValue(value=doc_id)
                                )
                            ]
                        )
                    )
                )
                deleted_count += 1

            return [TextContent(
                type="text",
                text=f"✅ Successfully deleted {deleted_count} documents from collection '{collection}'"
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error deleting documents: {str(e)}"
            )]

    async def _create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance_metric: str = "cosine"
    ) -> List[TextContent]:
        """Create a new vector collection."""
        try:
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT
            }

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance_metric, Distance.COSINE)
                )
            )

            return [TextContent(
                type="text",
                text=f"✅ Successfully created collection '{collection_name}' with {vector_size}D vectors using {distance_metric} distance"
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating collection: {str(e)}"
            )]

    async def _list_collections(self) -> List[TextContent]:
        """List all available collections."""
        try:
            collections = self.client.get_collections()

            if not collections.collections:
                return [TextContent(
                    type="text",
                    text="No collections found in the vector store"
                )]

            lines = ["📚 **Available Collections:**", ""]

            for collection in collections.collections:
                collection_info = self.client.get_collection(collection.name)
                lines.extend([
                    f"**{collection.name}**",
                    f"  - Vectors: {collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else 'N/A'}",
                    f"  - Points: {collection_info.points_count if hasattr(collection_info, 'points_count') else 'N/A'}",
                    f"  - Status: {collection_info.status if hasattr(collection_info, 'status') else 'Active'}",
                    ""
                ])

            return [TextContent(
                type="text",
                text="\n".join(lines)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error listing collections: {str(e)}"
            )]

    async def _ensure_collection_exists(self, collection_name: str):
        """Ensure a collection exists, create if it doesn't."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if collection_name not in collection_names:
                await self._create_collection(collection_name)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure collection exists: {e}")

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using OpenAI API."""
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        if not self.embedding_api_key:
            return None

        try:
            response = await self.embedding_client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.embedding_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.embedding_model,
                    "input": text
                }
            )
            response.raise_for_status()

            data = response.json()
            embedding = data["data"][0]["embedding"]

            # Cache the embedding
            self.embedding_cache[text] = embedding
            return embedding

        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)

                if boundary > start + chunk_size * 0.5:  # At least 50% of chunk size
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1

            chunks.append(chunk.strip())

            if end >= len(text):
                break

            start = end - chunk_overlap

        return chunks

    async def _get_collection_stats(self) -> str:
        """Get statistics for all collections."""
        try:
            collections = self.client.get_collections()
            stats = []

            for collection in collections.collections:
                collection_info = self.client.get_collection(collection.name)
                stats.append({
                    "name": collection.name,
                    "vectors_count": getattr(collection_info, 'vectors_count', 0),
                    "points_count": getattr(collection_info, 'points_count', 0),
                    "status": getattr(collection_info, 'status', 'unknown'),
                    "last_updated": datetime.now().isoformat()
                })

            return json.dumps(stats, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _get_health_status(self) -> str:
        """Get health status of the vector store."""
        try:
            # Try to get collections as health check
            collections = self.client.get_collections()

            health_info = {
                "status": "healthy",
                "qdrant_url": self.qdrant_url,
                "embedding_model": self.embedding_model,
                "collections_count": len(collections.collections),
                "cache_size": len(self.embedding_cache),
                "timestamp": datetime.now().isoformat()
            }

            return json.dumps(health_info, indent=2)

        except Exception as e:
            health_info = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(health_info, indent=2)

    async def setup_default_collections(self):
        """Set up default collections on startup."""
        for collection_name in self.default_collections:
            try:
                await self._ensure_collection_exists(collection_name)
            except Exception as e:
                print(f"Warning: Could not create default collection {collection_name}: {e}")

    async def run(self, transport_uri: str = "stdio://"):
        """Run the MCP server."""
        # Setup default collections
        await self.setup_default_collections()

        async with self.embedding_client:
            await self.server.run(
                transport_uri,
                InitializationOptions(
                    server_name="vector-store",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {},
                        "resources": {}
                    }
                )
            )


async def main():
    """Main entry point for the Vector Store MCP server."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    server = VectorStoreMCPServer(qdrant_url, embedding_dim)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())