"""MCP Server Manager.

Manages the lifecycle of multiple MCP servers for the Multi-Agent LangGraph system.
Provides centralized starting, stopping, and health monitoring of MCP servers.
"""

import asyncio
import logging
import subprocess
import signal
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager

from .config import MCPConfiguration, MCPServerConfig, MCPServerType, get_mcp_config


class MCPServerManager:
    """Manages multiple MCP servers for the Multi-Agent system."""

    def __init__(self, config: Optional[MCPConfiguration] = None):
        self.config = config or get_mcp_config()
        self.running_servers: Dict[str, subprocess.Popen] = {}
        self.server_health: Dict[str, bool] = {}
        self.logger = logging.getLogger(__name__)

        # Server module mapping
        self.server_modules = {
            MCPServerType.TAVILY_SEARCH: "mcp_servers.tavily_search",
            MCPServerType.VECTOR_STORE: "mcp_servers.vector_store",
            MCPServerType.DOCUMENT_PROCESSOR: "mcp_servers.document_processor",
            MCPServerType.ARXIV_RESEARCHER: "mcp_servers.arxiv_researcher",
            MCPServerType.WEB_RESEARCHER: "mcp_servers.web_researcher",
            MCPServerType.KNOWLEDGE_GRAPH: "mcp_servers.knowledge_graph",
            MCPServerType.WORKSPACE_MANAGER: "mcp_servers.workspace_manager",
        }

    async def start_server(self, server_config: MCPServerConfig) -> bool:
        """Start a single MCP server."""
        if server_config.name in self.running_servers:
            self.logger.warning(f"Server {server_config.name} is already running")
            return True

        try:
            # Get the server module
            module_name = self.server_modules.get(server_config.type)
            if not module_name:
                self.logger.error(f"Unknown server type: {server_config.type}")
                return False

            # Prepare environment variables
            env = os.environ.copy()
            if server_config.api_key_env_var:
                api_key = os.getenv(server_config.api_key_env_var)
                if not api_key:
                    self.logger.error(f"API key not found: {server_config.api_key_env_var}")
                    return False
                env[server_config.api_key_env_var] = api_key

            # Start the server process
            cmd = ["python", "-m", module_name]
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self.running_servers[server_config.name] = process
            self.server_health[server_config.name] = True

            self.logger.info(f"Started MCP server: {server_config.name} (PID: {process.pid})")

            # Wait a moment to check if the server started successfully
            await asyncio.sleep(1)
            if process.poll() is not None:
                # Process has already terminated
                stdout, stderr = process.communicate()
                self.logger.error(f"Server {server_config.name} failed to start:")
                self.logger.error(f"STDOUT: {stdout}")
                self.logger.error(f"STDERR: {stderr}")
                self.server_health[server_config.name] = False
                del self.running_servers[server_config.name]
                return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to start server {server_config.name}: {e}")
            self.server_health[server_config.name] = False
            return False

    async def stop_server(self, server_name: str) -> bool:
        """Stop a single MCP server."""
        if server_name not in self.running_servers:
            self.logger.warning(f"Server {server_name} is not running")
            return True

        try:
            process = self.running_servers[server_name]

            # Try graceful shutdown first
            process.terminate()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process(process)),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # Force kill if graceful shutdown fails
                self.logger.warning(f"Force killing server {server_name}")
                process.kill()
                await asyncio.create_task(self._wait_for_process(process))

            del self.running_servers[server_name]
            del self.server_health[server_name]

            self.logger.info(f"Stopped MCP server: {server_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop server {server_name}: {e}")
            return False

    async def start_all_servers(self) -> Dict[str, bool]:
        """Start all enabled MCP servers."""
        results = {}

        for server_config in self.config.get_enabled_servers():
            self.logger.info(f"Starting MCP server: {server_config.name}")
            success = await self.start_server(server_config)
            results[server_config.name] = success

            if not success:
                self.logger.error(f"Failed to start {server_config.name}")

        return results

    async def stop_all_servers(self) -> Dict[str, bool]:
        """Stop all running MCP servers."""
        results = {}

        for server_name in list(self.running_servers.keys()):
            success = await self.stop_server(server_name)
            results[server_name] = success

        return results

    async def restart_server(self, server_name: str) -> bool:
        """Restart a specific MCP server."""
        server_config = None
        for config in self.config.servers:
            if config.name == server_name:
                server_config = config
                break

        if not server_config:
            self.logger.error(f"Server configuration not found: {server_name}")
            return False

        await self.stop_server(server_name)
        return await self.start_server(server_config)

    async def health_check(self, server_name: str) -> bool:
        """Check if a server is healthy."""
        if server_name not in self.running_servers:
            return False

        process = self.running_servers[server_name]
        is_alive = process.poll() is None

        self.server_health[server_name] = is_alive

        if not is_alive:
            self.logger.warning(f"Server {server_name} is not responding")
            # Clean up dead process
            del self.running_servers[server_name]

        return is_alive

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all running servers."""
        results = {}

        for server_name in list(self.running_servers.keys()):
            results[server_name] = await self.health_check(server_name)

        return results

    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all servers."""
        status = {}

        for server_config in self.config.servers:
            server_name = server_config.name
            is_running = server_name in self.running_servers
            is_healthy = self.server_health.get(server_name, False)

            process_info = {}
            if is_running:
                process = self.running_servers[server_name]
                process_info = {
                    "pid": process.pid,
                    "return_code": process.poll()
                }

            status[server_name] = {
                "enabled": server_config.enabled,
                "type": server_config.type.value,
                "running": is_running,
                "healthy": is_healthy,
                "process": process_info
            }

        return status

    async def _wait_for_process(self, process: subprocess.Popen):
        """Wait for a process to terminate."""
        while process.poll() is None:
            await asyncio.sleep(0.1)

    @asynccontextmanager
    async def managed_servers(self):
        """Context manager for automatically managing server lifecycle."""
        try:
            # Start all servers
            start_results = await self.start_all_servers()
            failed_servers = [name for name, success in start_results.items() if not success]

            if failed_servers:
                self.logger.warning(f"Failed to start servers: {failed_servers}")

            yield self

        finally:
            # Stop all servers
            stop_results = await self.stop_all_servers()
            failed_stops = [name for name, success in stop_results.items() if not success]

            if failed_stops:
                self.logger.warning(f"Failed to stop servers: {failed_stops}")


# Global server manager instance
_server_manager: Optional[MCPServerManager] = None


def get_server_manager() -> MCPServerManager:
    """Get the global MCP server manager."""
    global _server_manager
    if _server_manager is None:
        _server_manager = MCPServerManager()
    return _server_manager


async def start_mcp_servers() -> Dict[str, bool]:
    """Start all MCP servers."""
    manager = get_server_manager()
    return await manager.start_all_servers()


async def stop_mcp_servers() -> Dict[str, bool]:
    """Stop all MCP servers."""
    manager = get_server_manager()
    return await manager.stop_all_servers()


def get_mcp_server_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all MCP servers."""
    manager = get_server_manager()
    return manager.get_server_status()


# CLI interface for server management
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server Manager")
    parser.add_argument("command", choices=["start", "stop", "restart", "status", "health"])
    parser.add_argument("--server", help="Specific server name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    async def main():
        manager = get_server_manager()

        if args.command == "start":
            if args.server:
                config = manager.config.get_server_config(MCPServerType(args.server))
                if config:
                    result = await manager.start_server(config)
                    print(f"Start {args.server}: {'Success' if result else 'Failed'}")
                else:
                    print(f"Server not found: {args.server}")
            else:
                results = await manager.start_all_servers()
                for server, success in results.items():
                    print(f"Start {server}: {'Success' if success else 'Failed'}")

        elif args.command == "stop":
            if args.server:
                result = await manager.stop_server(args.server)
                print(f"Stop {args.server}: {'Success' if result else 'Failed'}")
            else:
                results = await manager.stop_all_servers()
                for server, success in results.items():
                    print(f"Stop {server}: {'Success' if success else 'Failed'}")

        elif args.command == "restart":
            if args.server:
                result = await manager.restart_server(args.server)
                print(f"Restart {args.server}: {'Success' if result else 'Failed'}")
            else:
                await manager.stop_all_servers()
                await asyncio.sleep(2)
                results = await manager.start_all_servers()
                for server, success in results.items():
                    print(f"Restart {server}: {'Success' if success else 'Failed'}")

        elif args.command == "status":
            status = manager.get_server_status()
            for server, info in status.items():
                print(f"{server}: {info}")

        elif args.command == "health":
            health = await manager.health_check_all()
            for server, healthy in health.items():
                print(f"{server}: {'Healthy' if healthy else 'Unhealthy'}")

    asyncio.run(main())