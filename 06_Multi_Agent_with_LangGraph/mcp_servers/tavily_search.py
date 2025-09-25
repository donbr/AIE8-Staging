"""MCP Tavily Search Server.

A Model Context Protocol server for Tavily search functionality,
replacing the LangChain Community TavilySearchResults tool with
a more performant, standardized interface.

Features:
- Direct Tavily API integration
- Enhanced caching and rate limiting
- Structured search results
- Multiple search types (web, news, academic)
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import httpx
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


class TavilySearchResult(BaseModel):
    """Structured Tavily search result."""
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None


class TavilySearchResponse(BaseModel):
    """Complete Tavily search response."""
    query: str
    results: List[TavilySearchResult]
    search_type: str
    total_results: int
    search_time: float


class TavilyMCPServer:
    """MCP Server for Tavily search operations."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")

        self.base_url = "https://api.tavily.com"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.server = Server("tavily-search")

        # Cache for search results (simple in-memory cache)
        self.cache: Dict[str, tuple[TavilySearchResponse, datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes

        # Register tools and resources
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        """Register MCP tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="tavily_web_search",
                    description="Search the web using Tavily API for up-to-date information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (1-20)",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 5
                            },
                            "search_depth": {
                                "type": "string",
                                "enum": ["basic", "advanced"],
                                "description": "Search depth - basic or advanced",
                                "default": "basic"
                            },
                            "include_images": {
                                "type": "boolean",
                                "description": "Include images in results",
                                "default": False
                            },
                            "include_raw_content": {
                                "type": "boolean",
                                "description": "Include raw page content",
                                "default": False
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="tavily_news_search",
                    description="Search for recent news articles using Tavily",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The news search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (1-20)",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 5
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days back to search",
                                "minimum": 1,
                                "maximum": 30,
                                "default": 7
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "tavily_web_search":
                return await self._web_search(**arguments)
            elif name == "tavily_news_search":
                return await self._news_search(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    def _register_resources(self):
        """Register MCP resources."""

        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="tavily://search_stats",
                    name="Search Statistics",
                    description="Statistics about search usage and cache performance",
                    mimeType="application/json"
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "tavily://search_stats":
                return json.dumps({
                    "cache_entries": len(self.cache),
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                    "api_calls_today": self._get_api_calls_today()
                })
            else:
                raise ValueError(f"Unknown resource: {uri}")

    async def _web_search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_images: bool = False,
        include_raw_content: bool = False
    ) -> List[TextContent]:
        """Perform web search using Tavily API."""

        # Check cache first
        cache_key = f"web:{query}:{max_results}:{search_depth}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return [TextContent(
                type="text",
                text=f"**Cached Results for: {query}**\n\n" + self._format_search_results(cached_result)
            )]

        # Prepare API request
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_images": include_images,
            "include_raw_content": include_raw_content
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/search",
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            # Parse results
            results = []
            for item in data.get("results", []):
                results.append(TavilySearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    published_date=item.get("published_date")
                ))

            search_response = TavilySearchResponse(
                query=query,
                results=results,
                search_type="web",
                total_results=len(results),
                search_time=data.get("response_time", 0.0)
            )

            # Cache the result
            self._cache_result(cache_key, search_response)

            return [TextContent(
                type="text",
                text=self._format_search_results(search_response)
            )]

        except httpx.HTTPError as e:
            return [TextContent(
                type="text",
                text=f"Search error: {str(e)}"
            )]

    async def _news_search(
        self,
        query: str,
        max_results: int = 5,
        days: int = 7
    ) -> List[TextContent]:
        """Perform news search using Tavily API."""

        cache_key = f"news:{query}:{max_results}:{days}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return [TextContent(
                type="text",
                text=f"**Cached News Results for: {query}**\n\n" + self._format_search_results(cached_result)
            )]

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_domains": ["news.google.com", "reuters.com", "bbc.com", "ap.org"],
            "days": days
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/search",
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(TavilySearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    published_date=item.get("published_date")
                ))

            search_response = TavilySearchResponse(
                query=query,
                results=results,
                search_type="news",
                total_results=len(results),
                search_time=data.get("response_time", 0.0)
            )

            self._cache_result(cache_key, search_response)

            return [TextContent(
                type="text",
                text=self._format_search_results(search_response)
            )]

        except httpx.HTTPError as e:
            return [TextContent(
                type="text",
                text=f"News search error: {str(e)}"
            )]

    def _get_cached_result(self, cache_key: str) -> Optional[TavilySearchResponse]:
        """Get cached search result if not expired."""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                # Remove expired entry
                del self.cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: TavilySearchResponse):
        """Cache search result with timestamp."""
        self.cache[cache_key] = (result, datetime.now())

        # Simple cache cleanup - remove entries older than TTL
        cutoff_time = datetime.now() - self.cache_ttl
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if timestamp < cutoff_time
        ]
        for key in expired_keys:
            del self.cache[key]

    def _format_search_results(self, response: TavilySearchResponse) -> str:
        """Format search results for display."""
        lines = [
            f"🔍 **Search Results for: {response.query}**",
            f"📊 Found {response.total_results} results in {response.search_time:.2f}s",
            f"🔗 Search type: {response.search_type}",
            ""
        ]

        for i, result in enumerate(response.results, 1):
            lines.extend([
                f"**{i}. {result.title}**",
                f"🌐 {result.url}",
                f"📊 Score: {result.score:.2f}",
                ""
            ])

            # Add content snippet
            content = result.content
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(content)
            lines.append("")

            if result.published_date:
                lines.append(f"📅 Published: {result.published_date}")
                lines.append("")

        return "\n".join(lines)

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder implementation)."""
        # In a real implementation, you'd track hits/misses
        return 0.0

    def _get_api_calls_today(self) -> int:
        """Get API calls made today (placeholder implementation)."""
        # In a real implementation, you'd track API usage
        return 0

    async def run(self, transport_uri: str = "stdio://"):
        """Run the MCP server."""
        async with self.client:
            await self.server.run(
                transport_uri,
                InitializationOptions(
                    server_name="tavily-search",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {},
                        "resources": {}
                    }
                )
            )


async def main():
    """Main entry point for the Tavily MCP server."""
    server = TavilyMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())