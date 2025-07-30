
from typing import Dict, List, Optional, Any
import asyncio
import os
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
import shutil

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor


class Manus(ToolCallAgent):

    name: str = "Manus"
    description: str = "An advanced AI agent that can solve complex tasks using multiple tools including file analysis, GitHub integration, and web automation"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 15000  # Increased for better context
    max_steps: int = 1000     # Increased for complex tasks

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    _initialized: bool = False
    
    # Advanced API key management
    api_key_manager: Optional[object] = Field(default=None, exclude=True)
    current_key_config: Optional[Dict] = Field(default=None, exclude=True)
    retry_count: int = Field(default=0, exclude=True)

    # File and repository management
    uploaded_files: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    github_repos: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    workspace_files: List[str] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """Initialize basic components synchronously."""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, api_key_manager=None, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        # Use the advanced API key manager if provided
        if api_key_manager:
            # Get an available API key from the manager
            result = api_key_manager.get_available_api_key(use_random=True)
            if result:
                api_key, key_config = result
                kwargs['api_key'] = api_key
                kwargs['api_key_manager'] = api_key_manager
                kwargs['current_key_config'] = key_config
        
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        await instance.scan_workspace()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} with command {server_config.command}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def scan_workspace(self) -> None:
        """Scan workspace for files and repositories."""
        try:
            workspace_path = Path(config.workspace_root)
            if workspace_path.exists():
                # Scan for uploaded files
                uploads_path = workspace_path / "uploads"
                if uploads_path.exists():
                    for file_path in uploads_path.rglob("*"):
                        if file_path.is_file():
                            self.uploaded_files.append({
                                'name': file_path.name,
                                'path': str(file_path),
                                'size': file_path.stat().st_size,
                                'type': 'uploaded'
                            })

                # Scan for GitHub repositories
                github_path = workspace_path / "github_clones"
                if github_path.exists():
                    for repo_path in github_path.iterdir():
                        if repo_path.is_dir():
                            repo_files = list(repo_path.rglob("*"))
                            self.github_repos.append({
                                'name': repo_path.name,
                                'path': str(repo_path),
                                'files': [str(f) for f in repo_files if f.is_file()],
                                'type': 'github_repo'
                            })

                # Get all workspace files
                self.workspace_files = [str(f) for f in workspace_path.rglob("*") if f.is_file()]

        except Exception as e:
            logger.error(f"Error scanning workspace: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """Connect to an MCP server."""
        try:
            if use_stdio:
                if stdio_args:
                    command = [server_url] + stdio_args
                else:
                    command = [server_url]
                
                client = await self.mcp_clients.connect_stdio(command)
            else:
                client = await self.mcp_clients.connect_sse(server_url)
            
            if client:
                self.connected_servers[server_id] = server_url
                # Add MCP tools to available tools
                mcp_tool = MCPClientTool(client, server_id)
                self.available_tools.add_tool(mcp_tool)
                logger.info(f"Added MCP tools from server {server_id}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server."""
        try:
            if server_id in self.connected_servers:
                del self.connected_servers[server_id]
                # Remove MCP tools for this server
                tools_to_remove = [tool for tool in self.available_tools.tools if hasattr(tool, 'server_id') and tool.server_id == server_id]
                for tool in tools_to_remove:
                    self.available_tools.remove_tool(tool)
                logger.info(f"Disconnected from MCP server {server_id}")
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server {server_id}: {e}")

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Disconnect from all MCP servers
            for server_id in list(self.connected_servers.keys()):
                await self.disconnect_mcp_server(server_id)
            
            # Clean up browser context
            if self.browser_context_helper:
                await self.browser_context_helper.cleanup()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def think(self) -> bool:
        """Enhanced thinking process with file and repository context."""
        # Add context about available files and repositories
        context = self._build_context()
        
        # Update the system prompt with current context
        enhanced_prompt = self.system_prompt + "\n\n" + context
        
        # Call parent think method with enhanced context
        return await super().think()

    def _build_context(self) -> str:
        """Build context string with information about files and repositories."""
        context_parts = []
        
        if self.uploaded_files:
            context_parts.append("Uploaded Files:")
            for file_info in self.uploaded_files:
                context_parts.append(f"- {file_info['name']} ({file_info['size']} bytes)")
        
        if self.github_repos:
            context_parts.append("\nGitHub Repositories:")
            for repo_info in self.github_repos:
                context_parts.append(f"- {repo_info['name']} ({len(repo_info['files'])} files)")
                # Add some key files from the repository
                key_files = [f for f in repo_info['files'] if any(ext in f.lower() for ext in ['.py', '.js', '.html', '.md', 'readme', 'requirements'])]
                for key_file in key_files[:5]:  # Show first 5 key files
                    context_parts.append(f"  - {Path(key_file).name}")
        
        if self.workspace_files:
            context_parts.append(f"\nTotal workspace files: {len(self.workspace_files)}")
        
        return "\n".join(context_parts)

    async def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a file and return its content and metadata."""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {"error": "File not found"}
            
            # Get file metadata
            stat = file_path_obj.stat()
            file_info = {
                "name": file_path_obj.name,
                "path": str(file_path_obj),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": file_path_obj.suffix.lower(),
                "content": None,
                "type": "unknown"
            }
            
            # Determine file type and read content
            ext = file_path_obj.suffix.lower()
            if ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_info["content"] = content
                    file_info["type"] = "text"
                except UnicodeDecodeError:
                    file_info["type"] = "binary"
                    file_info["content"] = "Binary file - cannot display content"
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                file_info["type"] = "image"
            elif ext in ['.pdf']:
                file_info["type"] = "pdf"
            elif ext in ['.zip', '.rar', '.7z']:
                file_info["type"] = "archive"
            else:
                file_info["type"] = "unknown"
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return {"error": str(e)}

    async def clone_github_repo(self, repo_url: str, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """Clone a GitHub repository."""
        try:
            if not target_dir:
                # Create a temporary directory for the clone
                target_dir = tempfile.mkdtemp(prefix="github_clone_")
            
            # Extract repo name from URL
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            
            # Clone the repository
            result = subprocess.run(
                ['git', 'clone', repo_url, target_dir],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                return {"error": f"Failed to clone repository: {result.stderr}"}
            
            # Scan the cloned repository
            repo_path = Path(target_dir)
            repo_files = [str(f) for f in repo_path.rglob("*") if f.is_file()]
            
            repo_info = {
                "name": repo_name,
                "url": repo_url,
                "path": target_dir,
                "files": repo_files,
                "file_count": len(repo_files),
                "cloned_at": asyncio.get_event_loop().time()
            }
            
            # Add to tracked repositories
            self.github_repos.append(repo_info)
            
            return repo_info
            
        except subprocess.TimeoutExpired:
            return {"error": "Clone operation timed out"}
        except Exception as e:
            logger.error(f"Error cloning repository {repo_url}: {e}")
            return {"error": str(e)}

    async def create_zip_archive(self, files: List[str], output_path: str) -> Dict[str, Any]:
        """Create a ZIP archive from a list of files."""
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    if Path(file_path).exists():
                        # Add file to ZIP with relative path
                        arcname = Path(file_path).name
                        zipf.write(file_path, arcname)
            
            return {
                "success": True,
                "output_path": output_path,
                "file_count": len(files),
                "size": Path(output_path).stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error creating ZIP archive: {e}")
            return {"error": str(e)}

    async def search_files(self, query: str, file_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for files in the workspace based on query and file types."""
        try:
            results = []
            workspace_path = Path(config.workspace_root)
            
            # Define search patterns
            patterns = []
            if file_types:
                for ext in file_types:
                    patterns.append(f"**/*{ext}")
            else:
                patterns.append("**/*")
            
            for pattern in patterns:
                for file_path in workspace_path.glob(pattern):
                    if file_path.is_file():
                        # Check if file matches query
                        if query.lower() in file_path.name.lower():
                            file_info = await self.analyze_file(str(file_path))
                            if "error" not in file_info:
                                results.append(file_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []

    async def execute_code_analysis(self, file_path: str) -> Dict[str, Any]:
        """Perform code analysis on a file."""
        try:
            file_info = await self.analyze_file(file_path)
            if "error" in file_info:
                return file_info
            
            if file_info["type"] != "text":
                return {"error": "File is not a text file suitable for code analysis"}
            
            content = file_info["content"]
            ext = file_info["extension"]
            
            analysis = {
                "file_path": file_path,
                "file_name": file_info["name"],
                "file_size": file_info["size"],
                "language": self._detect_language(ext),
                "lines_of_code": len(content.split('\n')),
                "characters": len(content),
                "word_count": len(content.split()),
                "complexity_metrics": self._calculate_complexity(content, ext),
                "dependencies": self._extract_dependencies(content, ext),
                "functions": self._extract_functions(content, ext),
                "classes": self._extract_classes(content, ext)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing code in {file_path}: {e}")
            return {"error": str(e)}

    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.html': 'HTML',
            '.css': 'CSS',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'MATLAB',
            '.sh': 'Shell',
            '.sql': 'SQL',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.md': 'Markdown',
            '.txt': 'Text'
        }
        return language_map.get(extension, 'Unknown')

    def _calculate_complexity(self, content: str, extension: str) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        lines = content.split('\n')
        
        metrics = {
            "total_lines": len(lines),
            "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "average_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
        }
        
        # Language-specific metrics
        if extension == '.py':
            metrics.update(self._python_complexity(content))
        elif extension in ['.js', '.ts']:
            metrics.update(self._javascript_complexity(content))
        
        return metrics

    def _python_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate Python-specific complexity metrics."""
        lines = content.split('\n')
        
        # Count functions, classes, imports
        functions = len([line for line in lines if line.strip().startswith('def ')])
        classes = len([line for line in lines if line.strip().startswith('class ')])
        imports = len([line for line in lines if line.strip().startswith(('import ', 'from '))])
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "complexity_score": functions + classes * 2  # Simple complexity score
        }

    def _javascript_complexity(self, content: str) -> Dict[str, Any]:
        """Calculate JavaScript-specific complexity metrics."""
        lines = content.split('\n')
        
        # Count functions, classes, imports
        functions = len([line for line in lines if 'function ' in line or '=>' in line])
        classes = len([line for line in lines if 'class ' in line])
        imports = len([line for line in lines if line.strip().startswith(('import ', 'require('))])
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "complexity_score": functions + classes * 2
        }

    def _extract_dependencies(self, content: str, extension: str) -> List[str]:
        """Extract dependencies from code file."""
        dependencies = []
        lines = content.split('\n')
        
        if extension == '.py':
            # Python imports
            for line in lines:
                line = line.strip()
                if line.startswith(('import ', 'from ')):
                    # Extract module name
                    if line.startswith('import '):
                        module = line.split('import ')[1].split(' as ')[0].split()[0]
                        dependencies.append(module)
                    elif line.startswith('from '):
                        parts = line.split(' import ')
                        if len(parts) > 1:
                            module = parts[0].split('from ')[1].split('.')[0]
                            dependencies.append(module)
        
        elif extension in ['.js', '.ts']:
            # JavaScript/TypeScript imports
            for line in lines:
                line = line.strip()
                if line.startswith(('import ', 'require(')):
                    if line.startswith('import '):
                        # Extract from import statement
                        if ' from ' in line:
                            module = line.split(' from ')[1].replace("'", '').replace('"', '').split('/')[0]
                            dependencies.append(module)
                    elif line.startswith('require('):
                        # Extract from require statement
                        module = line.split('require(')[1].split(')')[0].replace("'", '').replace('"', '').split('/')[0]
                        dependencies.append(module)
        
        return list(set(dependencies))  # Remove duplicates

    def _extract_functions(self, content: str, extension: str) -> List[Dict[str, str]]:
        """Extract function definitions from code."""
        functions = []
        lines = content.split('\n')
        
        if extension == '.py':
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0]
                    functions.append({
                        "name": func_name,
                        "line": i + 1,
                        "signature": line
                    })
        
        elif extension in ['.js', '.ts']:
            for i, line in enumerate(lines):
                line = line.strip()
                if 'function ' in line or '=>' in line:
                    if 'function ' in line:
                        func_name = line.split('function ')[1].split('(')[0]
                        functions.append({
                            "name": func_name,
                            "line": i + 1,
                            "signature": line
                        })
                    elif '=>' in line and '=' in line:
                        func_name = line.split('=')[0].strip()
                        functions.append({
                            "name": func_name,
                            "line": i + 1,
                            "signature": line
                        })
        
        return functions

    def _extract_classes(self, content: str, extension: str) -> List[Dict[str, str]]:
        """Extract class definitions from code."""
        classes = []
        lines = content.split('\n')
        
        if extension == '.py':
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('class '):
                    class_name = line.split('class ')[1].split('(')[0].split(':')[0]
                    classes.append({
                        "name": class_name,
                        "line": i + 1,
                        "signature": line
                    })
        
        elif extension in ['.js', '.ts']:
            for i, line in enumerate(lines):
                line = line.strip()
                if 'class ' in line:
                    class_name = line.split('class ')[1].split(' ')[0].split('{')[0]
                    classes.append({
                        "name": class_name,
                        "line": i + 1,
                        "signature": line
                    })
        
        return classes

    async def generate_documentation(self, file_path: str) -> Dict[str, Any]:
        """Generate documentation for a code file."""
        try:
            analysis = await self.execute_code_analysis(file_path)
            if "error" in analysis:
                return analysis
            
            # Create documentation structure
            doc = {
                "file_path": file_path,
                "file_name": analysis["file_name"],
                "language": analysis["language"],
                "overview": f"This {analysis['language']} file contains {analysis['lines_of_code']} lines of code with {len(analysis['functions'])} functions and {len(analysis['classes'])} classes.",
                "dependencies": analysis["dependencies"],
                "functions": [],
                "classes": [],
                "usage_examples": [],
                "complexity_analysis": analysis["complexity_metrics"]
            }
            
            # Document functions
            for func in analysis["functions"]:
                doc["functions"].append({
                    "name": func["name"],
                    "line": func["line"],
                    "signature": func["signature"],
                    "description": f"Function {func['name']} defined at line {func['line']}"
                })
            
            # Document classes
            for cls in analysis["classes"]:
                doc["classes"].append({
                    "name": cls["name"],
                    "line": cls["line"],
                    "signature": cls["signature"],
                    "description": f"Class {cls['name']} defined at line {cls['line']}"
                })
            
            return doc
            
        except Exception as e:
            logger.error(f"Error generating documentation for {file_path}: {e}")
            return {"error": str(e)}

    async def run_tests(self, file_path: str) -> Dict[str, Any]:
        """Run tests for a file if available."""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {"error": "File not found"}
            
            ext = file_path_obj.suffix.lower()
            test_results = {
                "file_path": file_path,
                "tests_found": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "output": "",
                "error": None
            }
            
            if ext == '.py':
                # Look for test files
                test_file = file_path_obj.parent / f"test_{file_path_obj.name}"
                if test_file.exists():
                    test_results["tests_found"] = True
                    # Run pytest
                    result = subprocess.run(
                        ['python', '-m', 'pytest', str(test_file), '-v'],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    test_results["output"] = result.stdout + result.stderr
                    test_results["tests_run"] = len([line for line in result.stdout.split('\n') if '::' in line])
                    test_results["tests_passed"] = len([line for line in result.stdout.split('\n') if 'PASSED' in line])
                    test_results["tests_failed"] = len([line for line in result.stdout.split('\n') if 'FAILED' in line])
            
            return test_results
            
        except subprocess.TimeoutExpired:
            return {"error": "Test execution timed out"}
        except Exception as e:
            logger.error(f"Error running tests for {file_path}: {e}")
            return {"error": str(e)}

    async def suggest_improvements(self, file_path: str) -> Dict[str, Any]:
        """Suggest improvements for a code file."""
        try:
            analysis = await self.execute_code_analysis(file_path)
            if "error" in analysis:
                return analysis
            
            suggestions = {
                "file_path": file_path,
                "language": analysis["language"],
                "suggestions": [],
                "priority": "medium"
            }
            
            # Code quality suggestions
            metrics = analysis["complexity_metrics"]
            
            if metrics["average_line_length"] > 80:
                suggestions["suggestions"].append({
                    "type": "style",
                    "message": "Consider breaking long lines to improve readability",
                    "priority": "low"
                })
            
            if metrics["complexity_score"] > 10:
                suggestions["suggestions"].append({
                    "type": "complexity",
                    "message": "High complexity detected. Consider refactoring into smaller functions",
                    "priority": "high"
                })
            
            if len(analysis["dependencies"]) > 10:
                suggestions["suggestions"].append({
                    "type": "dependencies",
                    "message": "Many dependencies detected. Consider if all are necessary",
                    "priority": "medium"
                })
            
            if metrics["comment_lines"] < metrics["code_lines"] * 0.1:
                suggestions["suggestions"].append({
                    "type": "documentation",
                    "message": "Low comment ratio. Consider adding more documentation",
                    "priority": "medium"
                })
            
            # Set overall priority
            high_suggestions = len([s for s in suggestions["suggestions"] if s["priority"] == "high"])
            if high_suggestions > 0:
                suggestions["priority"] = "high"
            elif len(suggestions["suggestions"]) > 3:
                suggestions["priority"] = "medium"
            else:
                suggestions["priority"] = "low"
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting improvements for {file_path}: {e}")
            return {"error": str(e)}
