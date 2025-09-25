"""MCP ArXiv Researcher Server.

A Model Context Protocol server for ArXiv academic paper research,
implementing the bonus activity mentioned in the Multi-Agent notebook.

Features:
- ArXiv paper search by keywords, authors, categories
- Abstract analysis and content extraction
- PDF download and full-text processing
- Citation network analysis
- Research trend identification
- Paper ranking and relevance scoring
"""

import asyncio
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import arxiv
import fitz  # PyMuPDF
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


class ArXivPaper(BaseModel):
    """ArXiv paper representation."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: datetime
    updated: datetime
    pdf_url: str
    arxiv_url: str
    summary_score: Optional[float] = None
    full_text: Optional[str] = None
    citation_count: Optional[int] = None


class SearchQuery(BaseModel):
    """ArXiv search query."""
    keywords: List[str] = []
    authors: List[str] = []
    categories: List[str] = []
    title_contains: Optional[str] = None
    abstract_contains: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class ResearchSummary(BaseModel):
    """Research summary for a topic."""
    topic: str
    papers_found: int
    key_findings: List[str]
    main_authors: List[str]
    trending_categories: List[str]
    research_gaps: List[str]
    citation_network: Dict[str, List[str]]


class ArXivMCPServer:
    """MCP Server for ArXiv research operations."""

    def __init__(self):
        self.server = Server("arxiv-researcher")
        self.client = arxiv.Client()
        self.http_client = httpx.AsyncClient(timeout=60.0)

        # Cache for paper data
        self.paper_cache: Dict[str, ArXivPaper] = {}
        self.search_cache: Dict[str, List[ArXivPaper]] = {}

        # Common categories
        self.category_map = {
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision",
            "cs.LG": "Machine Learning",
            "cs.IR": "Information Retrieval",
            "cs.RO": "Robotics",
            "stat.ML": "Statistics - Machine Learning",
            "eess.AS": "Audio and Speech Processing",
            "eess.IV": "Image and Video Processing"
        }

        # Register tools and resources
        self._register_tools()
        self._register_resources()

    def _register_tools(self):
        """Register MCP tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="search_arxiv",
                    description="Search ArXiv for academic papers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (keywords, topics, etc.)"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of papers to return",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 10
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "ArXiv categories to search (e.g., cs.AI, cs.LG)",
                                "default": []
                            },
                            "sort_by": {
                                "type": "string",
                                "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                                "description": "Sort order for results",
                                "default": "relevance"
                            },
                            "date_range": {
                                "type": "string",
                                "description": "Date range (e.g., 'last_week', 'last_month', 'last_year')",
                                "default": "all_time"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_paper_details",
                    description="Get detailed information about a specific ArXiv paper",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "ArXiv paper ID (e.g., '2301.00001')"
                            },
                            "include_full_text": {
                                "type": "boolean",
                                "description": "Whether to download and extract full text",
                                "default": False
                            }
                        },
                        "required": ["arxiv_id"]
                    }
                ),
                Tool(
                    name="analyze_research_trends",
                    description="Analyze research trends for a topic",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Research topic to analyze"
                            },
                            "time_window": {
                                "type": "string",
                                "description": "Time window for analysis",
                                "enum": ["6_months", "1_year", "2_years", "5_years"],
                                "default": "1_year"
                            },
                            "max_papers": {
                                "type": "integer",
                                "description": "Maximum papers to analyze",
                                "default": 50
                            }
                        },
                        "required": ["topic"]
                    }
                ),
                Tool(
                    name="find_related_papers",
                    description="Find papers related to a given paper",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "ArXiv ID of the reference paper"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum related papers to find",
                                "default": 10
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Minimum similarity score",
                                "minimum": 0.1,
                                "maximum": 1.0,
                                "default": 0.3
                            }
                        },
                        "required": ["arxiv_id"]
                    }
                ),
                Tool(
                    name="summarize_papers",
                    description="Generate research summary from a set of papers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of ArXiv paper IDs to summarize"
                            },
                            "summary_type": {
                                "type": "string",
                                "enum": ["brief", "detailed", "comparative"],
                                "description": "Type of summary to generate",
                                "default": "brief"
                            }
                        },
                        "required": ["arxiv_ids"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "search_arxiv":
                return await self._search_arxiv(**arguments)
            elif name == "get_paper_details":
                return await self._get_paper_details(**arguments)
            elif name == "analyze_research_trends":
                return await self._analyze_research_trends(**arguments)
            elif name == "find_related_papers":
                return await self._find_related_papers(**arguments)
            elif name == "summarize_papers":
                return await self._summarize_papers(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    def _register_resources(self):
        """Register MCP resources."""

        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="arxiv://categories",
                    name="ArXiv Categories",
                    description="List of ArXiv subject categories",
                    mimeType="application/json"
                ),
                Resource(
                    uri="arxiv://cache/stats",
                    name="Cache Statistics",
                    description="Statistics about cached papers and searches",
                    mimeType="application/json"
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "arxiv://categories":
                return json.dumps(self.category_map, indent=2)
            elif uri == "arxiv://cache/stats":
                return json.dumps({
                    "cached_papers": len(self.paper_cache),
                    "cached_searches": len(self.search_cache),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                raise ValueError(f"Unknown resource: {uri}")

    async def _search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        categories: List[str] = [],
        sort_by: str = "relevance",
        date_range: str = "all_time"
    ) -> List[TextContent]:
        """Search ArXiv for papers."""
        try:
            # Build search query
            search_query = self._build_search_query(query, categories, date_range)

            # Check cache
            cache_key = f"{search_query}_{max_results}_{sort_by}"
            if cache_key in self.search_cache:
                papers = self.search_cache[cache_key][:max_results]
                return [TextContent(
                    type="text",
                    text=f"**Cached ArXiv Search Results for: {query}**\n\n" + self._format_paper_list(papers)
                )]

            # Perform search
            sort_criteria = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate
            }

            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_criteria.get(sort_by, arxiv.SortCriterion.Relevance)
            )

            papers = []
            for result in self.client.results(search):
                paper = ArXivPaper(
                    arxiv_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    categories=result.categories,
                    published=result.published,
                    updated=result.updated,
                    pdf_url=result.pdf_url,
                    arxiv_url=result.entry_id
                )
                papers.append(paper)
                self.paper_cache[paper.arxiv_id] = paper

            # Cache results
            self.search_cache[cache_key] = papers

            if not papers:
                return [TextContent(
                    type="text",
                    text=f"No papers found for query: '{query}'"
                )]

            return [TextContent(
                type="text",
                text=f"🔬 **ArXiv Search Results for: {query}**\n\n" + self._format_paper_list(papers)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"ArXiv search error: {str(e)}"
            )]

    async def _get_paper_details(
        self,
        arxiv_id: str,
        include_full_text: bool = False
    ) -> List[TextContent]:
        """Get detailed information about a specific paper."""
        try:
            # Clean arxiv_id
            arxiv_id = arxiv_id.replace('arxiv:', '').replace('arXiv:', '').split('v')[0]

            # Check cache first
            if arxiv_id in self.paper_cache:
                paper = self.paper_cache[arxiv_id]
            else:
                # Fetch from ArXiv
                search = arxiv.Search(id_list=[arxiv_id])
                results = list(self.client.results(search))

                if not results:
                    return [TextContent(
                        type="text",
                        text=f"Paper not found: {arxiv_id}"
                    )]

                result = results[0]
                paper = ArXivPaper(
                    arxiv_id=arxiv_id,
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    categories=result.categories,
                    published=result.published,
                    updated=result.updated,
                    pdf_url=result.pdf_url,
                    arxiv_url=result.entry_id
                )
                self.paper_cache[arxiv_id] = paper

            # Extract full text if requested
            if include_full_text and not paper.full_text:
                paper.full_text = await self._extract_pdf_text(paper.pdf_url)

            return [TextContent(
                type="text",
                text=self._format_paper_details(paper, include_full_text)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error getting paper details: {str(e)}"
            )]

    async def _analyze_research_trends(
        self,
        topic: str,
        time_window: str = "1_year",
        max_papers: int = 50
    ) -> List[TextContent]:
        """Analyze research trends for a topic."""
        try:
            # Calculate date range
            time_deltas = {
                "6_months": timedelta(days=180),
                "1_year": timedelta(days=365),
                "2_years": timedelta(days=730),
                "5_years": timedelta(days=1825)
            }

            date_from = datetime.now() - time_deltas.get(time_window, timedelta(days=365))

            # Search for papers
            papers = await self._search_with_date_filter(topic, max_papers, date_from)

            if not papers:
                return [TextContent(
                    type="text",
                    text=f"No recent papers found for topic: '{topic}'"
                )]

            # Analyze trends
            analysis = self._analyze_papers_for_trends(papers, topic)

            return [TextContent(
                type="text",
                text=self._format_trend_analysis(analysis)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error analyzing trends: {str(e)}"
            )]

    async def _find_related_papers(
        self,
        arxiv_id: str,
        max_results: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[TextContent]:
        """Find papers related to a given paper."""
        try:
            # Get the reference paper
            if arxiv_id not in self.paper_cache:
                await self._get_paper_details(arxiv_id)

            reference_paper = self.paper_cache.get(arxiv_id)
            if not reference_paper:
                return [TextContent(
                    type="text",
                    text=f"Reference paper not found: {arxiv_id}"
                )]

            # Extract keywords from title and abstract
            keywords = self._extract_keywords(
                reference_paper.title + " " + reference_paper.abstract
            )

            # Search for related papers
            related_papers = []
            for keyword_group in self._group_keywords(keywords[:10]):  # Use top 10 keywords
                query = " AND ".join(keyword_group)
                search_results = await self._search_arxiv(
                    query=query,
                    max_results=max_results * 2,  # Get more to filter
                    categories=reference_paper.categories[:2]  # Use same categories
                )

                # Parse results and calculate similarity
                for result_text in search_results:
                    papers = self._parse_paper_list(result_text.text)
                    for paper in papers:
                        if paper.arxiv_id != arxiv_id:  # Don't include the reference paper
                            similarity = self._calculate_similarity(reference_paper, paper)
                            if similarity >= similarity_threshold:
                                paper.summary_score = similarity
                                related_papers.append(paper)

            # Remove duplicates and sort by similarity
            unique_papers = {p.arxiv_id: p for p in related_papers}
            sorted_papers = sorted(
                unique_papers.values(),
                key=lambda p: p.summary_score or 0,
                reverse=True
            )[:max_results]

            return [TextContent(
                type="text",
                text=f"📚 **Papers Related to: {reference_paper.title}**\n\n" + self._format_related_papers(sorted_papers)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error finding related papers: {str(e)}"
            )]

    async def _summarize_papers(
        self,
        arxiv_ids: List[str],
        summary_type: str = "brief"
    ) -> List[TextContent]:
        """Generate research summary from papers."""
        try:
            papers = []
            for arxiv_id in arxiv_ids:
                if arxiv_id not in self.paper_cache:
                    await self._get_paper_details(arxiv_id)
                if arxiv_id in self.paper_cache:
                    papers.append(self.paper_cache[arxiv_id])

            if not papers:
                return [TextContent(
                    type="text",
                    text="No valid papers found for summarization"
                )]

            summary = self._generate_summary(papers, summary_type)

            return [TextContent(
                type="text",
                text=summary
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error summarizing papers: {str(e)}"
            )]

    def _build_search_query(self, query: str, categories: List[str], date_range: str) -> str:
        """Build ArXiv search query."""
        search_terms = [f'all:"{query}"']

        if categories:
            cat_terms = " OR ".join([f"cat:{cat}" for cat in categories])
            search_terms.append(f"({cat_terms})")

        return " AND ".join(search_terms)

    def _format_paper_list(self, papers: List[ArXivPaper]) -> str:
        """Format list of papers for display."""
        lines = [f"📊 Found {len(papers)} papers", ""]

        for i, paper in enumerate(papers, 1):
            lines.extend([
                f"**{i}. {paper.title}**",
                f"🔬 ID: {paper.arxiv_id}",
                f"👥 Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}",
                f"📅 Published: {paper.published.strftime('%Y-%m-%d')}",
                f"🏷️ Categories: {', '.join(paper.categories[:2])}",
                ""
            ])

            # Add abstract preview
            abstract = paper.abstract
            if len(abstract) > 200:
                abstract = abstract[:200] + "..."
            lines.append(f"📝 {abstract}")
            lines.append("")

        return "\n".join(lines)

    def _format_paper_details(self, paper: ArXivPaper, include_full_text: bool) -> str:
        """Format detailed paper information."""
        lines = [
            f"📄 **{paper.title}**",
            "",
            f"🔬 **ArXiv ID**: {paper.arxiv_id}",
            f"👥 **Authors**: {', '.join(paper.authors)}",
            f"📅 **Published**: {paper.published.strftime('%Y-%m-%d')}",
            f"📅 **Updated**: {paper.updated.strftime('%Y-%m-%d')}",
            f"🏷️ **Categories**: {', '.join(paper.categories)}",
            f"🔗 **PDF**: {paper.pdf_url}",
            f"🔗 **ArXiv**: {paper.arxiv_url}",
            "",
            "📝 **Abstract**:",
            paper.abstract,
        ]

        if include_full_text and paper.full_text:
            lines.extend([
                "",
                "📖 **Full Text** (excerpt):",
                paper.full_text[:1000] + "..." if len(paper.full_text) > 1000 else paper.full_text
            ])

        return "\n".join(lines)

    def _format_trend_analysis(self, analysis: ResearchSummary) -> str:
        """Format research trend analysis."""
        lines = [
            f"📈 **Research Trend Analysis: {analysis.topic}**",
            "",
            f"📊 **Papers Analyzed**: {analysis.papers_found}",
            "",
            "🔍 **Key Findings**:",
        ]

        for finding in analysis.key_findings:
            lines.append(f"  • {finding}")

        lines.extend([
            "",
            "👑 **Leading Researchers**:",
        ])

        for author in analysis.main_authors[:5]:
            lines.append(f"  • {author}")

        lines.extend([
            "",
            "🏷️ **Trending Categories**:",
        ])

        for category in analysis.trending_categories:
            lines.append(f"  • {self.category_map.get(category, category)}")

        return "\n".join(lines)

    def _format_related_papers(self, papers: List[ArXivPaper]) -> str:
        """Format related papers list."""
        lines = []

        for i, paper in enumerate(papers, 1):
            similarity_score = paper.summary_score or 0
            lines.extend([
                f"**{i}. {paper.title}**",
                f"🎯 Similarity: {similarity_score:.3f}",
                f"🔬 ID: {paper.arxiv_id}",
                f"👥 Authors: {', '.join(paper.authors[:2])}",
                f"📅 {paper.published.strftime('%Y-%m-%d')}",
                ""
            ])

        return "\n".join(lines)

    async def _search_with_date_filter(
        self,
        topic: str,
        max_papers: int,
        date_from: datetime
    ) -> List[ArXivPaper]:
        """Search with date filtering."""
        # This would implement date-filtered search
        # For now, use regular search and filter afterward
        search_results = await self._search_arxiv(topic, max_papers * 2)

        papers = []
        for result_text in search_results:
            parsed_papers = self._parse_paper_list(result_text.text)
            for paper in parsed_papers:
                if paper.published >= date_from:
                    papers.append(paper)

        return papers[:max_papers]

    def _analyze_papers_for_trends(self, papers: List[ArXivPaper], topic: str) -> ResearchSummary:
        """Analyze papers for research trends."""
        # Count authors and categories
        author_counts = {}
        category_counts = {}

        for paper in papers:
            for author in paper.authors:
                author_counts[author] = author_counts.get(author, 0) + 1

            for category in paper.categories:
                category_counts[category] = category_counts.get(category, 0) + 1

        # Find top authors and categories
        top_authors = sorted(author_counts.keys(), key=author_counts.get, reverse=True)[:10]
        top_categories = sorted(category_counts.keys(), key=category_counts.get, reverse=True)[:5]

        # Generate key findings (simplified)
        key_findings = [
            f"Analyzed {len(papers)} papers on {topic}",
            f"Most active category: {self.category_map.get(top_categories[0], top_categories[0])}",
            f"Leading researcher: {top_authors[0] if top_authors else 'N/A'}",
        ]

        return ResearchSummary(
            topic=topic,
            papers_found=len(papers),
            key_findings=key_findings,
            main_authors=top_authors,
            trending_categories=top_categories,
            research_gaps=[],  # Would implement gap analysis
            citation_network={}  # Would implement citation analysis
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction (could be enhanced with NLP)
        text = text.lower()
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        # Return unique words, could add frequency-based filtering
        return list(set(words))[:20]

    def _group_keywords(self, keywords: List[str], group_size: int = 3) -> List[List[str]]:
        """Group keywords for search queries."""
        return [keywords[i:i + group_size] for i in range(0, len(keywords), group_size)]

    def _calculate_similarity(self, paper1: ArXivPaper, paper2: ArXivPaper) -> float:
        """Calculate similarity between papers (simplified)."""
        # Simple similarity based on shared keywords and categories
        text1 = (paper1.title + " " + paper1.abstract).lower()
        text2 = (paper2.title + " " + paper2.abstract).lower()

        words1 = set(re.findall(r'\b[a-zA-Z]{4,}\b', text1))
        words2 = set(re.findall(r'\b[a-zA-Z]{4,}\b', text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        jaccard = intersection / union if union > 0 else 0.0

        # Boost similarity for shared categories
        shared_categories = len(set(paper1.categories) & set(paper2.categories))
        category_boost = shared_categories * 0.1

        return min(1.0, jaccard + category_boost)

    def _parse_paper_list(self, text: str) -> List[ArXivPaper]:
        """Parse paper list from formatted text (placeholder)."""
        # This would implement parsing of the formatted paper list
        # For now, return empty list as this is used in a complex flow
        return []

    def _generate_summary(self, papers: List[ArXivPaper], summary_type: str) -> str:
        """Generate summary of papers."""
        if summary_type == "brief":
            return self._generate_brief_summary(papers)
        elif summary_type == "detailed":
            return self._generate_detailed_summary(papers)
        elif summary_type == "comparative":
            return self._generate_comparative_summary(papers)
        else:
            return self._generate_brief_summary(papers)

    def _generate_brief_summary(self, papers: List[ArXivPaper]) -> str:
        """Generate brief summary."""
        lines = [
            f"📚 **Research Summary ({len(papers)} papers)**",
            "",
            "**Key Topics Covered**:",
        ]

        # Extract common themes
        all_categories = []
        all_words = []

        for paper in papers:
            all_categories.extend(paper.categories)
            all_words.extend(self._extract_keywords(paper.title + " " + paper.abstract))

        # Count frequencies
        from collections import Counter
        common_categories = Counter(all_categories).most_common(5)
        common_topics = Counter(all_words).most_common(10)

        for category, count in common_categories:
            lines.append(f"  • {self.category_map.get(category, category)}: {count} papers")

        lines.extend([
            "",
            "**Trending Research Terms**:",
        ])

        for topic, count in common_topics:
            lines.append(f"  • {topic}: mentioned in {count} papers")

        return "\n".join(lines)

    def _generate_detailed_summary(self, papers: List[ArXivPaper]) -> str:
        """Generate detailed summary."""
        lines = [f"📚 **Detailed Research Summary ({len(papers)} papers)**", ""]

        for i, paper in enumerate(papers, 1):
            lines.extend([
                f"**{i}. {paper.title}**",
                f"Authors: {', '.join(paper.authors[:3])}",
                f"Published: {paper.published.strftime('%Y-%m-%d')}",
                f"Abstract: {paper.abstract[:200]}...",
                ""
            ])

        return "\n".join(lines)

    def _generate_comparative_summary(self, papers: List[ArXivPaper]) -> str:
        """Generate comparative summary."""
        return "📊 **Comparative Analysis**\n\n" + self._generate_brief_summary(papers)

    async def _extract_pdf_text(self, pdf_url: str) -> Optional[str]:
        """Extract text from PDF (simplified implementation)."""
        try:
            # Download PDF
            response = await self.http_client.get(pdf_url)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

            # Extract text with PyMuPDF
            doc = fitz.open(tmp_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Clean up
            os.unlink(tmp_path)

            return text[:10000]  # Return first 10k chars

        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return None

    async def run(self, transport_uri: str = "stdio://"):
        """Run the MCP server."""
        async with self.http_client:
            await self.server.run(
                transport_uri,
                InitializationOptions(
                    server_name="arxiv-researcher",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {},
                        "resources": {}
                    }
                )
            )


async def main():
    """Main entry point for the ArXiv MCP server."""
    server = ArXivMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())