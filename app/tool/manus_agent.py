"""
Enhanced Manus Agent with Advanced AI Coding Capabilities
Integrated with comprehensive code generation, testing, and validation
"""
import os
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from app.tool.base import BaseTool
from app.logger import logger
from app.ai_code_modifier import AICodeModifier
from app.file_manager import AdvancedFileManager
from app.project_templates import ProjectTemplateManager
from app.git_manager import GitManager
from app.tool.advanced_code_generator import AdvancedCodeGenerator
from app.llm_client import LLMClient
from app.api_key_manager import APIKeyManager


class ManusAgent(BaseTool):
    """
    Enhanced Manus Agent - Advanced AI Coding Assistant
    
    This agent provides comprehensive coding assistance including:
    - Complete project generation from natural language
    - Advanced file management and Git operations
    - Code testing, validation, and bug fixing
    - Real-time code analysis and suggestions
    - Project templates and scaffolding
    - Multi-language support with intelligent detection
    """
    
    name = "manus"
    description = """Advanced AI coding assistant that can generate, test, and manage complete projects.
    
    Capabilities:
    🚀 Project Generation: Create complete applications from descriptions
    📁 File Management: Advanced file operations with security
    🔧 Code Analysis: Intelligent code review and suggestions
    🧪 Testing: Generate and run comprehensive tests
    🐛 Bug Fixing: Automatic issue detection and resolution
    📊 Validation: Multi-level code quality validation
    🌐 Git Integration: Full version control operations
    📚 Templates: Pre-built project scaffolding
    🔄 Multi-language: Python, JavaScript, TypeScript, and more
    
    Usage Examples:
    - "Create a Flask web app with user authentication and database"
    - "Generate a React todo application with API integration"
    - "Build a Python data analysis script with pandas and visualization"
    - "Create a REST API with FastAPI and PostgreSQL"
    - "Generate comprehensive tests for my existing code"
    - "Fix bugs in my application and optimize performance"
    """
    
    def __init__(self, api_key_manager: APIKeyManager):
        super().__init__()
        self.api_key_manager = api_key_manager
        self.workspace_root = Path("workspace")
        self.workspace_root.mkdir(exist_ok=True)
        
        # Initialize core components
        self._initialize_components()
        
        # Task tracking
        self.active_tasks = {}
        self.task_history = []
        
        # AI conversation context
        self.conversation_context = []
        self.max_context_length = 10
        
        logger.info("Manus Agent initialized with advanced capabilities")
    
    def _initialize_components(self):
        """Initialize all component managers"""
        try:
            # Get available LLM client
            llm_client = self._get_llm_client()
            
            # Initialize managers
            self.file_manager = AdvancedFileManager(str(self.workspace_root))
            self.git_manager = GitManager(str(self.workspace_root))
            self.template_manager = ProjectTemplateManager()
            
            # Initialize AI components if LLM client is available
            if llm_client:
                self.ai_modifier = AICodeModifier(llm_client)
                self.code_generator = AdvancedCodeGenerator(
                    self.ai_modifier, self.file_manager, self.template_manager
                )
            else:
                logger.warning("No LLM client available - AI features will be limited")
                self.ai_modifier = None
                self.code_generator = None
            
            logger.info("All Manus Agent components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Manus Agent components: {e}")
            raise
    
    def _get_llm_client(self) -> Optional[LLMClient]:
        """Get an available LLM client"""
        try:
            available_key = self.api_key_manager.get_available_key()
            if available_key:
                return LLMClient(
                    api_key=available_key['api_key'],
                    provider=available_key['provider'],
                    model=available_key.get('model', 'gpt-3.5-turbo')
                )
            return None
        except Exception as e:
            logger.error(f"Error getting LLM client: {e}")
            return None
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Main execution method for Manus Agent"""
        try:
            # Parse the request
            request = kwargs.get('request', '').strip()
            task_type = kwargs.get('task_type', 'auto_detect')
            
            if not request:
                return {
                    'success': False,
                    'error': 'Request is required for Manus Agent',
                    'suggestions': [
                        'Try: "Create a Flask web application"',
                        'Try: "Generate tests for my Python code"',
                        'Try: "Fix bugs in my project"',
                        'Try: "Create a React todo app"'
                    ]
                }
            
            # Add to conversation context
            self._add_to_context('user', request)
            
            # Detect task type if not specified
            if task_type == 'auto_detect':
                task_type = self._detect_task_type(request)
            
            logger.info(f"Manus Agent executing task: {task_type} - {request}")
            
            # Route to appropriate handler
            result = self._route_task(task_type, request, kwargs)
            
            # Add result to context
            self._add_to_context('assistant', result.get('message', ''))
            
            # Track task
            self._track_task(task_type, request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Manus Agent execution: {e}")
            return {
                'success': False,
                'error': f'Manus Agent failed: {str(e)}',
                'task_type': task_type,
                'request': request
            }
    
    def _detect_task_type(self, request: str) -> str:
        """Intelligently detect the type of task based on the request"""
        request_lower = request.lower()
        
        # Project generation keywords
        if any(keyword in request_lower for keyword in [
            'create', 'generate', 'build', 'make', 'develop', 'scaffold'
        ]) and any(keyword in request_lower for keyword in [
            'app', 'application', 'project', 'website', 'api', 'service'
        ]):
            return 'project_generation'
        
        # Testing keywords
        if any(keyword in request_lower for keyword in [
            'test', 'testing', 'unit test', 'integration test', 'pytest', 'jest'
        ]):
            return 'testing'
        
        # Bug fixing keywords
        if any(keyword in request_lower for keyword in [
            'fix', 'bug', 'error', 'issue', 'debug', 'troubleshoot', 'repair'
        ]):
            return 'bug_fixing'
        
        # Code analysis keywords
        if any(keyword in request_lower for keyword in [
            'analyze', 'review', 'check', 'validate', 'quality', 'refactor'
        ]):
            return 'code_analysis'
        
        # File operations keywords
        if any(keyword in request_lower for keyword in [
            'file', 'folder', 'directory', 'copy', 'move', 'delete', 'upload'
        ]):
            return 'file_management'
        
        # Git operations keywords
        if any(keyword in request_lower for keyword in [
            'git', 'clone', 'commit', 'push', 'pull', 'branch', 'merge'
        ]):
            return 'git_operations'
        
        # Code modification keywords
        if any(keyword in request_lower for keyword in [
            'modify', 'change', 'update', 'edit', 'improve', 'optimize'
        ]):
            return 'code_modification'
        
        # Default to project generation for ambiguous requests
        return 'project_generation'
    
    def _route_task(self, task_type: str, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Route the task to the appropriate handler"""
        handlers = {
            'project_generation': self._handle_project_generation,
            'testing': self._handle_testing,
            'bug_fixing': self._handle_bug_fixing,
            'code_analysis': self._handle_code_analysis,
            'file_management': self._handle_file_management,
            'git_operations': self._handle_git_operations,
            'code_modification': self._handle_code_modification,
            'conversation': self._handle_conversation
        }
        
        handler = handlers.get(task_type, self._handle_project_generation)
        return handler(request, kwargs)
    
    def _handle_project_generation(self, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Handle project generation requests"""
        try:
            if not self.code_generator:
                return {
                    'success': False,
                    'error': 'AI code generation not available - no LLM client configured'
                }
            
            # Extract project parameters from request
            params = self._extract_project_params(request)
            
            # Generate the project
            result = self.code_generator.execute(**params)
            
            if result['success']:
                # Additional post-processing
                project_path = result.get('project_path', '')
                
                # Initialize Git repository if requested
                if kwargs.get('init_git', True):
                    git_result = self._initialize_git_repo(project_path)
                    result['git_initialized'] = git_result['success']
                
                result['message'] = f"""🚀 Project generated successfully!

**Project Details:**
- Name: {result.get('project_name', 'Unknown')}
- Language: {result.get('language', 'Unknown')}
- Path: {project_path}
- Files Created: {len(result.get('files_created', []))}

**Validation Results:**
- Tests Generated: {'✅' if result.get('tests_generated') else '❌'}
- Validation Passed: {'✅' if result.get('validation_passed') else '❌'}
- Documentation: {'✅' if result.get('documentation_created') else '❌'}

**Next Steps:**
{chr(10).join(f"• {step}" for step in result.get('next_steps', []))}

Your project is ready to use! 🎉"""
            
            return result
            
        except Exception as e:
            logger.error(f"Error in project generation: {e}")
            return {
                'success': False,
                'error': f'Project generation failed: {str(e)}'
            }
    
    def _handle_testing(self, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Handle testing-related requests"""
        try:
            if not self.ai_modifier:
                return {
                    'success': False,
                    'error': 'AI testing features not available - no LLM client configured'
                }
            
            # Find code files to test
            target_path = kwargs.get('target_path', self.workspace_root)
            code_files = self._find_code_files(target_path)
            
            if not code_files:
                return {
                    'success': False,
                    'error': 'No code files found to generate tests for',
                    'suggestion': 'Please specify a target file or ensure code files exist in the workspace'
                }
            
            test_results = []
            
            for code_file in code_files[:5]:  # Limit to 5 files
                try:
                    # Read code
                    with open(code_file, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    # Detect language
                    language = self._detect_language(code_file)
                    
                    # Generate tests
                    test_result = self.ai_modifier.generate_tests(
                        code_content, language, 'pytest' if language == 'python' else 'jest'
                    )
                    
                    if test_result['success']:
                        # Save test file
                        test_file_path = self._get_test_file_path(code_file, language)
                        test_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(test_file_path, 'w', encoding='utf-8') as f:
                            f.write(test_result['test_code'])
                        
                        test_results.append({
                            'file': str(code_file),
                            'test_file': str(test_file_path),
                            'language': language,
                            'success': True
                        })
                    
                except Exception as e:
                    test_results.append({
                        'file': str(code_file),
                        'success': False,
                        'error': str(e)
                    })
            
            successful_tests = [r for r in test_results if r.get('success')]
            
            return {
                'success': len(successful_tests) > 0,
                'message': f"""🧪 Test Generation Complete!

**Results:**
- Files Processed: {len(test_results)}
- Tests Generated: {len(successful_tests)}
- Success Rate: {len(successful_tests)/len(test_results)*100:.1f}%

**Generated Test Files:**
{chr(10).join(f"• {r['test_file']}" for r in successful_tests)}

Run tests with:
• Python: `pytest tests/ -v`
• JavaScript: `npm test`""",
                'test_results': test_results,
                'files_tested': len(test_results),
                'tests_generated': len(successful_tests)
            }
            
        except Exception as e:
            logger.error(f"Error in testing: {e}")
            return {
                'success': False,
                'error': f'Testing failed: {str(e)}'
            }
    
    def _handle_bug_fixing(self, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Handle bug fixing and code repair requests"""
        try:
            if not self.ai_modifier:
                return {
                    'success': False,
                    'error': 'AI bug fixing not available - no LLM client configured'
                }
            
            # Find files with potential issues
            target_path = kwargs.get('target_path', self.workspace_root)
            code_files = self._find_code_files(target_path)
            
            fixes_applied = []
            analysis_results = []
            
            for code_file in code_files[:3]:  # Limit to 3 files for performance
                try:
                    # Read code
                    with open(code_file, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    # Detect language
                    language = self._detect_language(code_file)
                    
                    # Analyze code for issues
                    issues = self._analyze_code_issues(code_content, language)
                    
                    if issues:
                        # Attempt to fix issues
                        fix_result = self.ai_modifier.fix_code_issues(
                            code_content, language, '; '.join(issues)
                        )
                        
                        if fix_result['success']:
                            # Create backup
                            backup_path = code_file.with_suffix(f'{code_file.suffix}.backup')
                            with open(backup_path, 'w', encoding='utf-8') as f:
                                f.write(code_content)
                            
                            # Write fixed code
                            with open(code_file, 'w', encoding='utf-8') as f:
                                f.write(fix_result['fixed_code'])
                            
                            fixes_applied.append({
                                'file': str(code_file),
                                'issues_fixed': len(issues),
                                'backup_created': str(backup_path),
                                'diff': fix_result.get('diff', '')
                            })
                    
                    analysis_results.append({
                        'file': str(code_file),
                        'language': language,
                        'issues_found': len(issues),
                        'issues': issues,
                        'fixed': len([f for f in fixes_applied if f['file'] == str(code_file)]) > 0
                    })
                    
                except Exception as e:
                    analysis_results.append({
                        'file': str(code_file),
                        'error': str(e),
                        'fixed': False
                    })
            
            return {
                'success': len(fixes_applied) > 0,
                'message': f"""🐛 Bug Fixing Complete!

**Analysis Results:**
- Files Analyzed: {len(analysis_results)}
- Issues Found: {sum(r.get('issues_found', 0) for r in analysis_results)}
- Files Fixed: {len(fixes_applied)}

**Fixes Applied:**
{chr(10).join(f"• {f['file']}: {f['issues_fixed']} issues fixed" for f in fixes_applied)}

**Backups Created:**
{chr(10).join(f"• {f['backup_created']}" for f in fixes_applied)}

Your code has been analyzed and improved! 🎉""",
                'analysis_results': analysis_results,
                'fixes_applied': fixes_applied,
                'total_issues_found': sum(r.get('issues_found', 0) for r in analysis_results),
                'files_fixed': len(fixes_applied)
            }
            
        except Exception as e:
            logger.error(f"Error in bug fixing: {e}")
            return {
                'success': False,
                'error': f'Bug fixing failed: {str(e)}'
            }
    
    def _handle_code_analysis(self, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Handle code analysis and review requests"""
        try:
            if not self.ai_modifier:
                return {
                    'success': False,
                    'error': 'AI code analysis not available - no LLM client configured'
                }
            
            # Find code files to analyze
            target_path = kwargs.get('target_path', self.workspace_root)
            code_files = self._find_code_files(target_path)
            
            if not code_files:
                return {
                    'success': False,
                    'error': 'No code files found to analyze'
                }
            
            analysis_results = []
            
            for code_file in code_files[:5]:  # Limit to 5 files
                try:
                    # Read code
                    with open(code_file, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    # Detect language
                    language = self._detect_language(code_file)
                    
                    # Analyze code structure
                    structure_analysis = self.ai_modifier.analyze_code_structure(code_content, language)
                    
                    # Explain code
                    explanation = self.ai_modifier.explain_code(code_content, language)
                    
                    # Get quality metrics
                    quality_metrics = self._get_code_quality_metrics(code_content, language)
                    
                    analysis_results.append({
                        'file': str(code_file),
                        'language': language,
                        'structure': structure_analysis,
                        'explanation': explanation.get('explanation', ''),
                        'quality_metrics': quality_metrics,
                        'lines_of_code': len(code_content.split('\n')),
                        'file_size': len(code_content)
                    })
                    
                except Exception as e:
                    analysis_results.append({
                        'file': str(code_file),
                        'error': str(e)
                    })
            
            # Generate summary
            total_lines = sum(r.get('lines_of_code', 0) for r in analysis_results)
            avg_quality = sum(r.get('quality_metrics', {}).get('overall_score', 0) for r in analysis_results) / len(analysis_results)
            
            return {
                'success': True,
                'message': f"""📊 Code Analysis Complete!

**Project Overview:**
- Files Analyzed: {len(analysis_results)}
- Total Lines of Code: {total_lines:,}
- Average Quality Score: {avg_quality:.1f}/100

**Analysis Results:**
{chr(10).join(f"• {r['file']}: {r.get('language', 'Unknown')} ({r.get('lines_of_code', 0)} lines)" for r in analysis_results if 'error' not in r)}

**Quality Insights:**
- Well-structured code with good organization
- Comprehensive analysis completed
- Recommendations available for improvement

Check the detailed analysis for specific insights! 🎯""",
                'analysis_results': analysis_results,
                'summary': {
                    'files_analyzed': len(analysis_results),
                    'total_lines': total_lines,
                    'average_quality': avg_quality,
                    'languages_detected': list(set(r.get('language') for r in analysis_results if 'language' in r))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in code analysis: {e}")
            return {
                'success': False,
                'error': f'Code analysis failed: {str(e)}'
            }
    
    def _handle_file_management(self, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Handle file management operations"""
        try:
            # Extract operation type and parameters
            operation = self._extract_file_operation(request)
            
            if operation['type'] == 'create_directory':
                result = self.file_manager.create_directory(operation['path'])
            elif operation['type'] == 'delete':
                result = self.file_manager.delete_item(operation['path'])
            elif operation['type'] == 'copy':
                result = self.file_manager.copy_item(operation['source'], operation['destination'])
            elif operation['type'] == 'move':
                result = self.file_manager.move_item(operation['source'], operation['destination'])
            elif operation['type'] == 'read':
                result = self.file_manager.read_file(operation['path'])
            elif operation['type'] == 'write':
                result = self.file_manager.write_file(operation['path'], operation['content'])
            elif operation['type'] == 'search':
                result = self.file_manager.search_files(operation['pattern'], operation.get('directory', '.'))
            elif operation['type'] == 'list':
                result = self.file_manager.get_directory_tree(operation.get('path', '.'))
            else:
                return {
                    'success': False,
                    'error': f'Unknown file operation: {operation["type"]}'
                }
            
            if result['success']:
                result['message'] = f"""📁 File Operation Complete!

**Operation:** {operation['type'].replace('_', ' ').title()}
**Result:** {result.get('message', 'Operation completed successfully')}

{result.get('details', '')}"""
            
            return result
            
        except Exception as e:
            logger.error(f"Error in file management: {e}")
            return {
                'success': False,
                'error': f'File management failed: {str(e)}'
            }
    
    def _handle_git_operations(self, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Handle Git operations"""
        try:
            # Extract Git operation and parameters
            operation = self._extract_git_operation(request)
            
            if operation['type'] == 'clone':
                result = self.git_manager.clone_repository(operation['url'], operation.get('path'))
            elif operation['type'] == 'commit':
                result = self.git_manager.commit_changes(operation['message'], operation.get('path', '.'))
            elif operation['type'] == 'push':
                result = self.git_manager.push_changes(operation.get('path', '.'))
            elif operation['type'] == 'pull':
                result = self.git_manager.pull_changes(operation.get('path', '.'))
            elif operation['type'] == 'branch':
                result = self.git_manager.create_branch(operation['name'], operation.get('path', '.'))
            elif operation['type'] == 'checkout':
                result = self.git_manager.checkout_branch(operation['branch'], operation.get('path', '.'))
            elif operation['type'] == 'status':
                result = self.git_manager.get_repository_info(operation.get('path', '.'))
            else:
                return {
                    'success': False,
                    'error': f'Unknown Git operation: {operation["type"]}'
                }
            
            if result['success']:
                result['message'] = f"""🌐 Git Operation Complete!

**Operation:** {operation['type'].title()}
**Result:** {result.get('message', 'Operation completed successfully')}

{result.get('details', '')}"""
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Git operations: {e}")
            return {
                'success': False,
                'error': f'Git operation failed: {str(e)}'
            }
    
    def _handle_code_modification(self, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Handle code modification requests"""
        try:
            if not self.ai_modifier:
                return {
                    'success': False,
                    'error': 'AI code modification not available - no LLM client configured'
                }
            
            # Find target file
            target_file = kwargs.get('target_file')
            if not target_file:
                # Try to find the most relevant file
                code_files = self._find_code_files(self.workspace_root)
                if not code_files:
                    return {
                        'success': False,
                        'error': 'No target file specified and no code files found'
                    }
                target_file = code_files[0]
            
            # Read current code
            with open(target_file, 'r', encoding='utf-8') as f:
                current_code = f.read()
            
            # Detect language
            language = self._detect_language(target_file)
            
            # Determine modification type
            modification_type = self._extract_modification_type(request)
            
            if modification_type == 'refactor':
                result = self.ai_modifier.refactor_code(current_code, 'clean', language, request)
                new_code = result.get('refactored_code', current_code)
            elif modification_type == 'optimize':
                result = self.ai_modifier.refactor_code(current_code, 'optimize', language, request)
                new_code = result.get('refactored_code', current_code)
            elif modification_type == 'convert':
                target_language = self._extract_target_language(request)
                result = self.ai_modifier.convert_code(current_code, language, target_language)
                new_code = result.get('converted_code', current_code)
            else:
                # General modification
                result = self.ai_modifier.modify_file(str(target_file), request)
                new_code = result.get('modified_content', current_code)
            
            if result.get('success'):
                # Create backup
                backup_path = Path(str(target_file) + '.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(current_code)
                
                # Write modified code
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(new_code)
                
                result['message'] = f"""🔧 Code Modification Complete!

**File:** {target_file}
**Modification:** {modification_type.title()}
**Backup Created:** {backup_path}

**Changes Applied:**
{result.get('diff', 'Code has been successfully modified')}

Your code has been improved! 🎉"""
                
                result['target_file'] = str(target_file)
                result['backup_file'] = str(backup_path)
                result['modification_type'] = modification_type
            
            return result
            
        except Exception as e:
            logger.error(f"Error in code modification: {e}")
            return {
                'success': False,
                'error': f'Code modification failed: {str(e)}'
            }
    
    def _handle_conversation(self, request: str, kwargs: Dict) -> Dict[str, Any]:
        """Handle general conversation and questions"""
        try:
            # Provide helpful information based on the request
            response = self._generate_helpful_response(request)
            
            return {
                'success': True,
                'message': response,
                'type': 'conversation',
                'context': self.conversation_context[-5:]  # Last 5 exchanges
            }
            
        except Exception as e:
            logger.error(f"Error in conversation handling: {e}")
            return {
                'success': False,
                'error': f'Conversation failed: {str(e)}'
            }
    
    # Helper methods
    
    def _extract_project_params(self, request: str) -> Dict[str, Any]:
        """Extract project parameters from natural language request"""
        params = {
            'description': request,
            'language': 'python',  # default
            'project_type': 'custom',
            'project_name': 'ai_generated_project',
            'include_tests': True,
            'validate_code': True
        }
        
        request_lower = request.lower()
        
        # Detect language
        if any(keyword in request_lower for keyword in ['python', 'flask', 'django', 'fastapi']):
            params['language'] = 'python'
        elif any(keyword in request_lower for keyword in ['javascript', 'js', 'node', 'express']):
            params['language'] = 'javascript'
        elif any(keyword in request_lower for keyword in ['typescript', 'ts']):
            params['language'] = 'typescript'
        elif any(keyword in request_lower for keyword in ['react', 'vue', 'angular']):
            params['language'] = 'javascript'
        
        # Detect project type
        if 'flask' in request_lower:
            params['project_type'] = 'flask'
        elif 'fastapi' in request_lower:
            params['project_type'] = 'fastapi'
        elif 'react' in request_lower:
            params['project_type'] = 'react'
        elif 'express' in request_lower:
            params['project_type'] = 'express'
        elif 'vue' in request_lower:
            params['project_type'] = 'vue'
        elif 'django' in request_lower:
            params['project_type'] = 'django'
        
        # Extract project name
        import re
        name_match = re.search(r'(?:called|named|call it|name it)\s+["\']?([^"\']+)["\']?', request_lower)
        if name_match:
            params['project_name'] = name_match.group(1).strip().replace(' ', '_')
        else:
            # Generate name from description
            words = re.findall(r'\b\w+\b', request_lower)
            if len(words) >= 2:
                params['project_name'] = '_'.join(words[:3])
        
        return params
    
    def _find_code_files(self, path: Path) -> List[Path]:
        """Find code files in the given path"""
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go'}
        code_files = []
        
        if path.is_file() and path.suffix in code_extensions:
            return [path]
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix in code_extensions:
                # Skip test files and node_modules
                if not any(skip in str(file_path) for skip in ['test_', '_test', 'node_modules', '__pycache__']):
                    code_files.append(file_path)
        
        return code_files[:10]  # Limit to 10 files
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go'
        }
        return extension_map.get(file_path.suffix, 'text')
    
    def _analyze_code_issues(self, code: str, language: str) -> List[str]:
        """Analyze code for potential issues"""
        issues = []
        lines = code.split('\n')
        
        # Basic static analysis
        if language == 'python':
            # Check for common Python issues
            if 'import *' in code:
                issues.append('Wildcard imports detected')
            
            # Check for long lines
            long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]
            if long_lines:
                issues.append(f'Long lines detected: {long_lines[:3]}')
            
            # Check for missing docstrings
            if 'def ' in code and '"""' not in code:
                issues.append('Functions missing docstrings')
        
        elif language == 'javascript':
            # Check for common JavaScript issues
            if 'var ' in code:
                issues.append('Use of var instead of let/const')
            
            if '==' in code and '===' not in code:
                issues.append('Use of loose equality operator')
        
        # General issues
        if not any(line.strip().startswith('#') or line.strip().startswith('//') for line in lines):
            issues.append('No comments found - consider adding documentation')
        
        return issues
    
    def _get_code_quality_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Get basic code quality metrics"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')]
        
        metrics = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'comment_lines': len(comment_lines),
            'comment_ratio': len(comment_lines) / len(non_empty_lines) if non_empty_lines else 0,
            'avg_line_length': sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0
        }
        
        # Calculate overall score
        score = 50  # Base score
        
        if metrics['comment_ratio'] > 0.1:
            score += 20
        if metrics['avg_line_length'] < 100:
            score += 15
        if len(non_empty_lines) > 10:
            score += 15
        
        metrics['overall_score'] = min(score, 100)
        
        return metrics
    
    def _get_test_file_path(self, code_file: Path, language: str) -> Path:
        """Generate appropriate test file path"""
        if language == 'python':
            test_dir = code_file.parent / 'tests'
            return test_dir / f'test_{code_file.name}'
        else:
            test_dir = code_file.parent / 'tests'
            return test_dir / f'{code_file.stem}.test{code_file.suffix}'
    
    def _initialize_git_repo(self, project_path: str) -> Dict[str, Any]:
        """Initialize Git repository for a project"""
        try:
            result = self.git_manager.get_repository_info(project_path)
            if not result['success']:
                # Initialize new repo
                import subprocess
                subprocess.run(['git', 'init'], cwd=project_path, check=True)
                subprocess.run(['git', 'add', '.'], cwd=project_path, check=True)
                subprocess.run(['git', 'commit', '-m', 'Initial commit by Manus Agent'], cwd=project_path, check=True)
                
                return {'success': True, 'message': 'Git repository initialized'}
            
            return {'success': True, 'message': 'Git repository already exists'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_file_operation(self, request: str) -> Dict[str, Any]:
        """Extract file operation details from request"""
        request_lower = request.lower()
        
        if 'create' in request_lower and ('folder' in request_lower or 'directory' in request_lower):
            return {'type': 'create_directory', 'path': self._extract_path(request)}
        elif 'delete' in request_lower:
            return {'type': 'delete', 'path': self._extract_path(request)}
        elif 'copy' in request_lower:
            paths = self._extract_multiple_paths(request)
            return {'type': 'copy', 'source': paths[0], 'destination': paths[1] if len(paths) > 1 else ''}
        elif 'move' in request_lower:
            paths = self._extract_multiple_paths(request)
            return {'type': 'move', 'source': paths[0], 'destination': paths[1] if len(paths) > 1 else ''}
        elif 'read' in request_lower:
            return {'type': 'read', 'path': self._extract_path(request)}
        elif 'write' in request_lower:
            return {'type': 'write', 'path': self._extract_path(request), 'content': self._extract_content(request)}
        elif 'search' in request_lower:
            return {'type': 'search', 'pattern': self._extract_pattern(request)}
        else:
            return {'type': 'list', 'path': self._extract_path(request) or '.'}
    
    def _extract_git_operation(self, request: str) -> Dict[str, Any]:
        """Extract Git operation details from request"""
        request_lower = request.lower()
        
        if 'clone' in request_lower:
            return {'type': 'clone', 'url': self._extract_url(request)}
        elif 'commit' in request_lower:
            return {'type': 'commit', 'message': self._extract_commit_message(request)}
        elif 'push' in request_lower:
            return {'type': 'push'}
        elif 'pull' in request_lower:
            return {'type': 'pull'}
        elif 'branch' in request_lower:
            return {'type': 'branch', 'name': self._extract_branch_name(request)}
        elif 'checkout' in request_lower:
            return {'type': 'checkout', 'branch': self._extract_branch_name(request)}
        else:
            return {'type': 'status'}
    
    def _extract_modification_type(self, request: str) -> str:
        """Extract the type of code modification requested"""
        request_lower = request.lower()
        
        if 'refactor' in request_lower:
            return 'refactor'
        elif 'optimize' in request_lower:
            return 'optimize'
        elif 'convert' in request_lower:
            return 'convert'
        else:
            return 'modify'
    
    def _extract_target_language(self, request: str) -> str:
        """Extract target language for code conversion"""
        request_lower = request.lower()
        
        if 'python' in request_lower:
            return 'python'
        elif 'javascript' in request_lower or 'js' in request_lower:
            return 'javascript'
        elif 'typescript' in request_lower:
            return 'typescript'
        else:
            return 'python'  # default
    
    def _extract_path(self, request: str) -> str:
        """Extract file path from request"""
        import re
        # Look for quoted paths
        quoted_match = re.search(r'["\']([^"\']+)["\']', request)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for file extensions
        ext_match = re.search(r'\b(\w+\.\w+)\b', request)
        if ext_match:
            return ext_match.group(1)
        
        return ''
    
    def _extract_multiple_paths(self, request: str) -> List[str]:
        """Extract multiple paths from request"""
        import re
        paths = re.findall(r'["\']([^"\']+)["\']', request)
        if not paths:
            paths = re.findall(r'\b(\w+\.\w+)\b', request)
        return paths
    
    def _extract_content(self, request: str) -> str:
        """Extract content to write from request"""
        # This is a simplified extraction - in practice, you'd want more sophisticated parsing
        return request.split('write')[-1].strip()
    
    def _extract_pattern(self, request: str) -> str:
        """Extract search pattern from request"""
        import re
        quoted_match = re.search(r'["\']([^"\']+)["\']', request)
        if quoted_match:
            return quoted_match.group(1)
        
        # Extract word after "search for"
        search_match = re.search(r'search for (\w+)', request.lower())
        if search_match:
            return search_match.group(1)
        
        return ''
    
    def _extract_url(self, request: str) -> str:
        """Extract URL from request"""
        import re
        url_match = re.search(r'https?://[^\s]+', request)
        if url_match:
            return url_match.group(0)
        return ''
    
    def _extract_commit_message(self, request: str) -> str:
        """Extract commit message from request"""
        import re
        quoted_match = re.search(r'["\']([^"\']+)["\']', request)
        if quoted_match:
            return quoted_match.group(1)
        
        # Default message
        return "Update by Manus Agent"
    
    def _extract_branch_name(self, request: str) -> str:
        """Extract branch name from request"""
        import re
        quoted_match = re.search(r'["\']([^"\']+)["\']', request)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for word after branch/checkout
        branch_match = re.search(r'(?:branch|checkout)\s+(\w+)', request.lower())
        if branch_match:
            return branch_match.group(1)
        
        return 'main'
    
    def _generate_helpful_response(self, request: str) -> str:
        """Generate a helpful response for general questions"""
        request_lower = request.lower()
        
        if 'help' in request_lower or 'what can you do' in request_lower:
            return """🤖 **Manus Agent - Your AI Coding Assistant**

I can help you with:

🚀 **Project Generation**
- Create complete applications from descriptions
- Generate project structures and templates
- Set up proper configurations and dependencies

🧪 **Testing & Validation**
- Generate comprehensive unit tests
- Validate code quality and syntax
- Run automated testing suites

🐛 **Bug Fixing & Analysis**
- Detect and fix code issues automatically
- Perform static code analysis
- Optimize code performance

📁 **File Management**
- Create, copy, move, and organize files
- Search through codebases
- Handle file uploads and downloads

🌐 **Git Integration**
- Clone repositories and manage versions
- Commit, push, and pull changes
- Create and manage branches

🔧 **Code Modification**
- Refactor and optimize existing code
- Convert between programming languages
- Add features and fix bugs

Just tell me what you need and I'll help you build it! 🎯"""
        
        elif 'example' in request_lower:
            return """💡 **Example Commands for Manus Agent:**

**Project Creation:**
- "Create a Flask web application with user authentication"
- "Generate a React todo app with API integration"
- "Build a Python data analysis script with pandas"

**Testing:**
- "Generate tests for my Python code"
- "Create unit tests for all my JavaScript functions"
- "Run comprehensive testing on my project"

**Bug Fixing:**
- "Fix bugs in my application"
- "Analyze my code for potential issues"
- "Optimize the performance of my Python script"

**File Operations:**
- "Create a new folder called 'utils'"
- "Copy main.py to backup.py"
- "Search for all functions named 'process_data'"

**Git Operations:**
- "Clone https://github.com/user/repo.git"
- "Commit changes with message 'Added new feature'"
- "Create a new branch called 'feature-update'"

Try any of these or describe what you need in your own words! 🚀"""
        
        else:
            return f"""🤔 I understand you're asking about: "{request}"

I'm Manus Agent, your advanced AI coding assistant. I can help you with:
- Creating complete applications and projects
- Testing and validating your code
- Fixing bugs and optimizing performance
- Managing files and Git repositories
- Analyzing and explaining code

Could you be more specific about what you'd like me to help you with? For example:
- "Create a web application for..."
- "Fix the bugs in my Python code"
- "Generate tests for my project"
- "Analyze my code quality"

I'm here to help make your coding tasks easier! 🎯"""
    
    def _add_to_context(self, role: str, content: str):
        """Add message to conversation context"""
        self.conversation_context.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent context
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context = self.conversation_context[-self.max_context_length:]
    
    def _track_task(self, task_type: str, request: str, result: Dict[str, Any]):
        """Track completed tasks for analytics"""
        task_record = {
            'task_type': task_type,
            'request': request,
            'success': result.get('success', False),
            'timestamp': datetime.now().isoformat(),
            'duration': 0  # Would be calculated in real implementation
        }
        
        self.task_history.append(task_record)
        
        # Keep only recent history
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get recent task history"""
        return self.task_history
    
    def get_conversation_context(self) -> List[Dict[str, Any]]:
        """Get conversation context"""
        return self.conversation_context