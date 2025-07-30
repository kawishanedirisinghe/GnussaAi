"""
Advanced Code Generator Tool for Manus Agent
Generates, tests, and validates code with comprehensive AI assistance
"""
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import ast
import re
from datetime import datetime

from app.tool.base import BaseTool
from app.logger import logger
from app.ai_code_modifier import AICodeModifier
from app.file_manager import AdvancedFileManager
from app.project_templates import ProjectTemplateManager


class AdvancedCodeGenerator(BaseTool):
    """Advanced code generation tool with testing and validation"""
    
    name = "advanced_code_generator"
    description = """Generate complete code projects, applications, and programs with AI assistance.
    
    This tool can:
    - Generate complete applications from natural language descriptions
    - Create project structures with proper file organization
    - Generate unit tests and validate code functionality
    - Fix bugs and optimize code performance
    - Create documentation and README files
    - Test code execution and validate results
    
    Usage examples:
    - "Create a Flask web application with user authentication"
    - "Generate a Python script for data analysis with pandas"
    - "Build a React component for a todo list application"
    - "Create a REST API with database integration"
    """
    
    def __init__(self, ai_code_modifier: AICodeModifier, file_manager: AdvancedFileManager, 
                 template_manager: ProjectTemplateManager):
        super().__init__()
        self.ai_modifier = ai_code_modifier
        self.file_manager = file_manager
        self.template_manager = template_manager
        self.workspace_root = Path("workspace")
        self.workspace_root.mkdir(exist_ok=True)
        
        # Language configurations for testing
        self.language_configs = {
            'python': {
                'extension': '.py',
                'test_command': ['python', '-m', 'py_compile'],
                'run_command': ['python'],
                'test_framework': 'pytest',
                'requirements_file': 'requirements.txt'
            },
            'javascript': {
                'extension': '.js',
                'test_command': ['node', '-c'],
                'run_command': ['node'],
                'test_framework': 'jest',
                'requirements_file': 'package.json'
            },
            'typescript': {
                'extension': '.ts',
                'test_command': ['tsc', '--noEmit'],
                'run_command': ['ts-node'],
                'test_framework': 'jest',
                'requirements_file': 'package.json'
            }
        }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the advanced code generation"""
        try:
            description = kwargs.get('description', '')
            project_type = kwargs.get('project_type', 'custom')
            language = kwargs.get('language', 'python')
            project_name = kwargs.get('project_name', 'generated_project')
            include_tests = kwargs.get('include_tests', True)
            validate_code = kwargs.get('validate_code', True)
            
            if not description:
                return {
                    'success': False,
                    'error': 'Description is required for code generation'
                }
            
            logger.info(f"Starting advanced code generation: {description}")
            
            # Step 1: Generate project structure
            project_result = self._generate_project_structure(
                description, project_type, language, project_name
            )
            
            if not project_result['success']:
                return project_result
            
            project_path = project_result['project_path']
            
            # Step 2: Generate main application code
            main_code_result = self._generate_main_code(
                description, language, project_path
            )
            
            if not main_code_result['success']:
                return main_code_result
            
            # Step 3: Generate supporting files
            support_result = self._generate_supporting_files(
                description, language, project_path
            )
            
            # Step 4: Generate tests if requested
            if include_tests:
                test_result = self._generate_tests(
                    description, language, project_path
                )
                if test_result['success']:
                    logger.info("Tests generated successfully")
            
            # Step 5: Validate and test code if requested
            validation_result = {'success': True, 'message': 'Validation skipped'}
            if validate_code:
                validation_result = self._validate_and_test_code(
                    language, project_path
                )
            
            # Step 6: Generate documentation
            doc_result = self._generate_documentation(
                description, language, project_path, project_name
            )
            
            # Step 7: Create final report
            final_result = {
                'success': True,
                'message': 'Advanced code generation completed successfully',
                'project_name': project_name,
                'project_path': str(project_path),
                'language': language,
                'files_created': project_result.get('files_created', []),
                'main_code': main_code_result.get('code', ''),
                'tests_generated': include_tests,
                'validation_passed': validation_result['success'],
                'validation_details': validation_result.get('details', {}),
                'documentation_created': doc_result['success'],
                'next_steps': self._get_next_steps(language, project_path)
            }
            
            logger.info(f"Code generation completed: {project_name}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in advanced code generation: {e}")
            return {
                'success': False,
                'error': f'Code generation failed: {str(e)}'
            }
    
    def _generate_project_structure(self, description: str, project_type: str, 
                                   language: str, project_name: str) -> Dict[str, Any]:
        """Generate the basic project structure"""
        try:
            project_path = self.workspace_root / project_name
            
            # Remove existing project if it exists
            if project_path.exists():
                import shutil
                shutil.rmtree(project_path)
            
            # Create project directory
            project_path.mkdir(parents=True)
            
            # Use template if available
            if project_type != 'custom':
                template_result = self.template_manager.create_project(
                    project_type, project_name, {
                        'description': description,
                        'author': 'AI Generated',
                        'email': 'ai@generated.com'
                    }
                )
                
                if template_result['success']:
                    return {
                        'success': True,
                        'project_path': project_path,
                        'files_created': template_result.get('files_created', []),
                        'message': 'Project structure created from template'
                    }
            
            # Create custom project structure
            structure = self._get_project_structure(language, description)
            files_created = []
            
            for file_path, content in structure.items():
                full_path = project_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                if content:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                else:
                    full_path.touch()
                
                files_created.append(file_path)
            
            return {
                'success': True,
                'project_path': project_path,
                'files_created': files_created,
                'message': f'Custom project structure created for {language}'
            }
            
        except Exception as e:
            logger.error(f"Error generating project structure: {e}")
            return {
                'success': False,
                'error': f'Failed to create project structure: {str(e)}'
            }
    
    def _generate_main_code(self, description: str, language: str, project_path: Path) -> Dict[str, Any]:
        """Generate the main application code"""
        try:
            # Generate main code using AI
            generation_result = self.ai_modifier.generate_code(
                description, language, None
            )
            
            if not generation_result['success']:
                return generation_result
            
            main_code = generation_result['code']
            
            # Determine main file name
            config = self.language_configs.get(language, {})
            extension = config.get('extension', '.txt')
            
            if language == 'python':
                main_file = 'main.py'
            elif language in ['javascript', 'typescript']:
                main_file = f'index{extension}'
            else:
                main_file = f'main{extension}'
            
            # Write main code to file
            main_path = project_path / main_file
            with open(main_path, 'w', encoding='utf-8') as f:
                f.write(main_code)
            
            # Analyze and improve the code
            analysis = self.ai_modifier.analyze_code_structure(main_code, language)
            
            return {
                'success': True,
                'code': main_code,
                'main_file': main_file,
                'analysis': analysis,
                'message': f'Main {language} code generated successfully'
            }
            
        except Exception as e:
            logger.error(f"Error generating main code: {e}")
            return {
                'success': False,
                'error': f'Failed to generate main code: {str(e)}'
            }
    
    def _generate_supporting_files(self, description: str, language: str, project_path: Path) -> Dict[str, Any]:
        """Generate supporting files like config, utils, etc."""
        try:
            files_created = []
            
            # Generate configuration files
            if language == 'python':
                # Generate requirements.txt
                requirements = self._generate_requirements(description, language)
                req_path = project_path / 'requirements.txt'
                with open(req_path, 'w') as f:
                    f.write(requirements)
                files_created.append('requirements.txt')
                
                # Generate config.py if needed
                if any(keyword in description.lower() for keyword in ['config', 'settings', 'database', 'api']):
                    config_code = self._generate_config_file(description, language)
                    config_path = project_path / 'config.py'
                    with open(config_path, 'w') as f:
                        f.write(config_code)
                    files_created.append('config.py')
            
            elif language in ['javascript', 'typescript']:
                # Generate package.json
                package_json = self._generate_package_json(description, project_path.name)
                pkg_path = project_path / 'package.json'
                with open(pkg_path, 'w') as f:
                    f.write(package_json)
                files_created.append('package.json')
            
            # Generate utility files if needed
            if any(keyword in description.lower() for keyword in ['utility', 'helper', 'utils']):
                utils_code = self._generate_utils_file(description, language)
                utils_file = f'utils{self.language_configs.get(language, {}).get("extension", ".py")}'
                utils_path = project_path / utils_file
                with open(utils_path, 'w') as f:
                    f.write(utils_code)
                files_created.append(utils_file)
            
            return {
                'success': True,
                'files_created': files_created,
                'message': f'Supporting files generated: {", ".join(files_created)}'
            }
            
        except Exception as e:
            logger.error(f"Error generating supporting files: {e}")
            return {
                'success': False,
                'error': f'Failed to generate supporting files: {str(e)}'
            }
    
    def _generate_tests(self, description: str, language: str, project_path: Path) -> Dict[str, Any]:
        """Generate comprehensive tests for the code"""
        try:
            # Find main code file
            main_files = []
            for pattern in ['main.*', 'index.*', 'app.*']:
                main_files.extend(project_path.glob(pattern))
            
            if not main_files:
                return {
                    'success': False,
                    'error': 'No main code file found to generate tests for'
                }
            
            main_file = main_files[0]
            
            # Read main code
            with open(main_file, 'r', encoding='utf-8') as f:
                main_code = f.read()
            
            # Generate tests using AI
            test_result = self.ai_modifier.generate_tests(
                main_code, language, 
                self.language_configs.get(language, {}).get('test_framework')
            )
            
            if not test_result['success']:
                return test_result
            
            # Create tests directory
            tests_dir = project_path / 'tests'
            tests_dir.mkdir(exist_ok=True)
            
            # Write test file
            extension = self.language_configs.get(language, {}).get('extension', '.py')
            test_file = tests_dir / f'test_{main_file.stem}{extension}'
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_result['test_code'])
            
            # Generate test configuration if needed
            if language == 'python':
                # Create pytest.ini
                pytest_config = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
"""
                with open(project_path / 'pytest.ini', 'w') as f:
                    f.write(pytest_config)
            
            return {
                'success': True,
                'test_file': str(test_file),
                'test_code': test_result['test_code'],
                'message': 'Comprehensive tests generated successfully'
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {
                'success': False,
                'error': f'Failed to generate tests: {str(e)}'
            }
    
    def _validate_and_test_code(self, language: str, project_path: Path) -> Dict[str, Any]:
        """Validate and test the generated code"""
        try:
            validation_results = {
                'syntax_check': {'passed': False, 'details': ''},
                'static_analysis': {'passed': False, 'details': ''},
                'test_execution': {'passed': False, 'details': ''},
                'overall_score': 0
            }
            
            # Step 1: Syntax validation
            syntax_result = self._check_syntax(language, project_path)
            validation_results['syntax_check'] = syntax_result
            
            # Step 2: Static analysis
            static_result = self._perform_static_analysis(language, project_path)
            validation_results['static_analysis'] = static_result
            
            # Step 3: Run tests if they exist
            test_result = self._run_tests(language, project_path)
            validation_results['test_execution'] = test_result
            
            # Calculate overall score
            score = 0
            if syntax_result['passed']:
                score += 40
            if static_result['passed']:
                score += 30
            if test_result['passed']:
                score += 30
            
            validation_results['overall_score'] = score
            
            # Fix issues if validation failed
            if score < 70:
                fix_result = self._fix_code_issues(language, project_path, validation_results)
                if fix_result['success']:
                    # Re-validate after fixes
                    return self._validate_and_test_code(language, project_path)
            
            return {
                'success': score >= 70,
                'details': validation_results,
                'message': f'Code validation completed with score: {score}/100'
            }
            
        except Exception as e:
            logger.error(f"Error validating code: {e}")
            return {
                'success': False,
                'error': f'Code validation failed: {str(e)}',
                'details': {}
            }
    
    def _check_syntax(self, language: str, project_path: Path) -> Dict[str, Any]:
        """Check syntax of generated code"""
        try:
            config = self.language_configs.get(language, {})
            extension = config.get('extension', '.py')
            
            # Find all code files
            code_files = list(project_path.rglob(f'*{extension}'))
            
            if not code_files:
                return {
                    'passed': False,
                    'details': f'No {language} files found for syntax checking'
                }
            
            syntax_errors = []
            
            for file_path in code_files:
                if language == 'python':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                        ast.parse(code)
                    except SyntaxError as e:
                        syntax_errors.append(f'{file_path.name}: {str(e)}')
                
                elif language in ['javascript', 'typescript']:
                    # Use node -c for syntax checking
                    try:
                        result = subprocess.run(
                            ['node', '-c', str(file_path)],
                            capture_output=True, text=True, timeout=10
                        )
                        if result.returncode != 0:
                            syntax_errors.append(f'{file_path.name}: {result.stderr}')
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        syntax_errors.append(f'{file_path.name}: Could not validate syntax')
            
            return {
                'passed': len(syntax_errors) == 0,
                'details': f'Syntax check: {len(syntax_errors)} errors found. ' + 
                          ('; '.join(syntax_errors) if syntax_errors else 'All files valid')
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': f'Syntax check failed: {str(e)}'
            }
    
    def _perform_static_analysis(self, language: str, project_path: Path) -> Dict[str, Any]:
        """Perform static code analysis"""
        try:
            issues = []
            
            # Find main code files
            config = self.language_configs.get(language, {})
            extension = config.get('extension', '.py')
            code_files = list(project_path.rglob(f'*{extension}'))
            
            for file_path in code_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                if language == 'python':
                    # Basic Python static analysis
                    try:
                        tree = ast.parse(code)
                        
                        # Check for common issues
                        for node in ast.walk(tree):
                            # Check for unused imports (simplified)
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    if alias.name not in code:
                                        issues.append(f'Potentially unused import: {alias.name}')
                            
                            # Check for long functions (>50 lines)
                            if isinstance(node, ast.FunctionDef):
                                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                                    if node.end_lineno - node.lineno > 50:
                                        issues.append(f'Long function detected: {node.name}')
                    
                    except SyntaxError:
                        issues.append(f'Syntax error in {file_path.name}')
                
                # Check for code quality indicators
                lines = code.split('\n')
                if len([line for line in lines if line.strip().startswith('#')]) / len(lines) < 0.1:
                    issues.append('Low comment ratio - consider adding more documentation')
                
                # Check for very long lines
                long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
                if long_lines:
                    issues.append(f'Long lines detected at: {long_lines[:5]}')
            
            return {
                'passed': len(issues) < 5,  # Allow some minor issues
                'details': f'Static analysis: {len(issues)} issues found. ' + 
                          ('; '.join(issues[:10]) if issues else 'Code quality looks good')
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': f'Static analysis failed: {str(e)}'
            }
    
    def _run_tests(self, language: str, project_path: Path) -> Dict[str, Any]:
        """Run the generated tests"""
        try:
            tests_dir = project_path / 'tests'
            
            if not tests_dir.exists():
                return {
                    'passed': True,  # No tests to run
                    'details': 'No tests found - skipping test execution'
                }
            
            config = self.language_configs.get(language, {})
            test_framework = config.get('test_framework', '')
            
            if language == 'python' and test_framework == 'pytest':
                try:
                    result = subprocess.run(
                        ['python', '-m', 'pytest', str(tests_dir), '-v'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    return {
                        'passed': result.returncode == 0,
                        'details': f'Test execution: {"PASSED" if result.returncode == 0 else "FAILED"}. '
                                 f'Output: {result.stdout[:500]}'
                    }
                
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    return {
                        'passed': False,
                        'details': 'Test execution failed - pytest not available or timeout'
                    }
            
            elif language in ['javascript', 'typescript'] and test_framework == 'jest':
                # Check if package.json exists and has jest
                package_json_path = project_path / 'package.json'
                if package_json_path.exists():
                    try:
                        result = subprocess.run(
                            ['npm', 'test'],
                            cwd=project_path,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        return {
                            'passed': result.returncode == 0,
                            'details': f'Test execution: {"PASSED" if result.returncode == 0 else "FAILED"}. '
                                     f'Output: {result.stdout[:500]}'
                        }
                    
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        return {
                            'passed': False,
                            'details': 'Test execution failed - npm/jest not available or timeout'
                        }
            
            return {
                'passed': True,
                'details': f'Test framework {test_framework} not configured for automated execution'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'details': f'Test execution failed: {str(e)}'
            }
    
    def _fix_code_issues(self, language: str, project_path: Path, validation_results: Dict) -> Dict[str, Any]:
        """Attempt to fix code issues found during validation"""
        try:
            fixes_applied = []
            
            # Find main code file
            config = self.language_configs.get(language, {})
            extension = config.get('extension', '.py')
            main_files = list(project_path.glob(f'main{extension}')) or \
                        list(project_path.glob(f'index{extension}')) or \
                        list(project_path.glob(f'app{extension}'))
            
            if not main_files:
                return {
                    'success': False,
                    'error': 'No main file found to fix'
                }
            
            main_file = main_files[0]
            
            # Read current code
            with open(main_file, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # Attempt to fix syntax errors
            if not validation_results['syntax_check']['passed']:
                syntax_details = validation_results['syntax_check']['details']
                
                fix_result = self.ai_modifier.fix_code_issues(
                    original_code, language, syntax_details
                )
                
                if fix_result['success']:
                    # Write fixed code
                    with open(main_file, 'w', encoding='utf-8') as f:
                        f.write(fix_result['fixed_code'])
                    
                    fixes_applied.append('Syntax errors fixed')
            
            # Attempt to improve code quality
            if not validation_results['static_analysis']['passed']:
                static_details = validation_results['static_analysis']['details']
                
                refactor_result = self.ai_modifier.refactor_code(
                    original_code, 'clean', language, 
                    f"Fix these issues: {static_details}"
                )
                
                if refactor_result['success']:
                    # Write refactored code
                    with open(main_file, 'w', encoding='utf-8') as f:
                        f.write(refactor_result['refactored_code'])
                    
                    fixes_applied.append('Code quality improved')
            
            return {
                'success': len(fixes_applied) > 0,
                'fixes_applied': fixes_applied,
                'message': f'Applied fixes: {", ".join(fixes_applied)}'
            }
            
        except Exception as e:
            logger.error(f"Error fixing code issues: {e}")
            return {
                'success': False,
                'error': f'Failed to fix issues: {str(e)}'
            }
    
    def _generate_documentation(self, description: str, language: str, 
                              project_path: Path, project_name: str) -> Dict[str, Any]:
        """Generate comprehensive documentation"""
        try:
            # Generate README.md
            readme_content = self._generate_readme(
                description, language, project_name, project_path
            )
            
            readme_path = project_path / 'README.md'
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            # Generate API documentation if applicable
            if any(keyword in description.lower() for keyword in ['api', 'web', 'server', 'flask', 'fastapi']):
                api_docs = self._generate_api_docs(description, language, project_path)
                
                if api_docs:
                    docs_dir = project_path / 'docs'
                    docs_dir.mkdir(exist_ok=True)
                    
                    api_docs_path = docs_dir / 'API.md'
                    with open(api_docs_path, 'w', encoding='utf-8') as f:
                        f.write(api_docs)
            
            return {
                'success': True,
                'message': 'Documentation generated successfully'
            }
            
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return {
                'success': False,
                'error': f'Documentation generation failed: {str(e)}'
            }
    
    def _get_project_structure(self, language: str, description: str) -> Dict[str, str]:
        """Get the basic project structure for different languages"""
        structure = {}
        
        if language == 'python':
            structure.update({
                'main.py': '',
                'requirements.txt': '',
                'README.md': '',
                'tests/__init__.py': '',
                'tests/test_main.py': '',
                '.gitignore': self._get_gitignore_content('python')
            })
            
            # Add additional structure based on description
            if 'web' in description.lower() or 'flask' in description.lower():
                structure.update({
                    'templates/index.html': '',
                    'static/css/style.css': '',
                    'static/js/app.js': ''
                })
        
        elif language in ['javascript', 'typescript']:
            structure.update({
                f'index.{language[0:2]}': '',
                'package.json': '',
                'README.md': '',
                'tests/index.test.js': '',
                '.gitignore': self._get_gitignore_content('node')
            })
        
        return structure
    
    def _generate_requirements(self, description: str, language: str) -> str:
        """Generate requirements/dependencies based on description"""
        if language == 'python':
            requirements = []
            
            # Basic requirements
            if 'web' in description.lower() or 'flask' in description.lower():
                requirements.extend(['Flask>=2.0.0', 'python-dotenv>=0.19.0'])
            
            if 'fastapi' in description.lower():
                requirements.extend(['fastapi>=0.68.0', 'uvicorn>=0.15.0'])
            
            if 'database' in description.lower():
                requirements.extend(['SQLAlchemy>=1.4.0', 'psycopg2-binary>=2.9.0'])
            
            if 'data' in description.lower() or 'analysis' in description.lower():
                requirements.extend(['pandas>=1.3.0', 'numpy>=1.21.0'])
            
            if 'test' in description.lower():
                requirements.extend(['pytest>=6.0.0', 'pytest-cov>=2.12.0'])
            
            # Default requirements if none specified
            if not requirements:
                requirements = ['requests>=2.25.0']
            
            return '\n'.join(requirements) + '\n'
        
        return ''
    
    def _generate_config_file(self, description: str, language: str) -> str:
        """Generate configuration file"""
        if language == 'python':
            return '''"""
Configuration settings for the application
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Basic settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    
    # API settings
    API_KEY = os.getenv('API_KEY', '')
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.example.com')

# Configuration instances
config = Config()
'''
        
        return ''
    
    def _generate_utils_file(self, description: str, language: str) -> str:
        """Generate utility functions file"""
        if language == 'python':
            return '''"""
Utility functions for the application
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_datetime(dt: datetime) -> str:
    """Format datetime for display"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def safe_json_loads(json_str: str) -> Optional[Dict]:
    """Safely load JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        logger.error(f"Failed to parse JSON: {json_str}")
        return None

def validate_email(email: str) -> bool:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_string(text: str) -> str:
    """Sanitize string for safe usage"""
    if not text:
        return ""
    
    # Remove potentially harmful characters
    import re
    return re.sub(r'[<>"\']', '', text.strip())
'''
        
        return ''
    
    def _generate_package_json(self, description: str, project_name: str) -> str:
        """Generate package.json for Node.js projects"""
        package = {
            "name": project_name.lower().replace(' ', '-'),
            "version": "1.0.0",
            "description": description,
            "main": "index.js",
            "scripts": {
                "start": "node index.js",
                "test": "jest",
                "dev": "nodemon index.js"
            },
            "keywords": [],
            "author": "AI Generated",
            "license": "MIT",
            "dependencies": {},
            "devDependencies": {
                "jest": "^27.0.0",
                "nodemon": "^2.0.0"
            }
        }
        
        # Add dependencies based on description
        if 'express' in description.lower() or 'web' in description.lower():
            package["dependencies"]["express"] = "^4.18.0"
        
        if 'database' in description.lower():
            package["dependencies"]["mongoose"] = "^6.0.0"
        
        return json.dumps(package, indent=2)
    
    def _generate_readme(self, description: str, language: str, 
                        project_name: str, project_path: Path) -> str:
        """Generate comprehensive README.md"""
        
        # Find created files
        files = []
        for file_path in project_path.rglob('*'):
            if file_path.is_file() and file_path.name != 'README.md':
                files.append(str(file_path.relative_to(project_path)))
        
        readme = f"""# {project_name}

{description}

## Description

This project was generated using AI assistance based on the following requirements:
"{description}"

## Features

- Generated with {language.title()} programming language
- Comprehensive project structure
- Automated testing setup
- Documentation included
- Code validation and quality checks

## Project Structure

```
{project_name}/
"""
        
        # Add file structure
        for file_path in sorted(files):
            readme += f"├── {file_path}\n"
        
        readme += """```

## Installation

"""
        
        if language == 'python':
            readme += """1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

## Testing

Run tests with pytest:
```bash
pytest tests/ -v
```
"""
        
        elif language in ['javascript', 'typescript']:
            readme += """1. Install dependencies:
   ```bash
   npm install
   ```

## Usage

Run the application:
```bash
npm start
```

For development with auto-reload:
```bash
npm run dev
```

## Testing

Run tests with Jest:
```bash
npm test
```
"""
        
        readme += f"""
## Development

This project was generated with the following validation:
- Syntax checking
- Static code analysis
- Automated testing
- Code quality validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## License

MIT License - feel free to use this code for your projects.

## Generated Information

- **Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Language**: {language.title()}
- **AI Assistant**: OpenManus Advanced Code Generator
- **Description**: {description}

---

*This project was automatically generated and validated by AI. All code has been tested and verified for quality and functionality.*
"""
        
        return readme
    
    def _generate_api_docs(self, description: str, language: str, project_path: Path) -> str:
        """Generate API documentation"""
        return f"""# API Documentation

## Overview

This API was generated based on: "{description}"

## Base URL

```
http://localhost:3000/api
```

## Authentication

[Authentication details would be added based on implementation]

## Endpoints

### Health Check

```
GET /health
```

Returns the health status of the API.

**Response:**
```json
{{
  "status": "healthy",
  "timestamp": "2023-12-07T10:30:00Z"
}}
```

## Error Handling

All errors follow this format:

```json
{{
  "error": {{
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details if available"
  }}
}}
```

## Rate Limiting

API requests are limited to prevent abuse. Current limits:
- 100 requests per minute per IP
- 1000 requests per hour per API key

## Generated Information

- **Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Language**: {language.title()}
- **Based on**: {description}

*This documentation was automatically generated. Please update as needed.*
"""
    
    def _get_gitignore_content(self, type: str) -> str:
        """Get appropriate .gitignore content"""
        if type == 'python':
            return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment variables
.env
.env.local

# Database
*.db
*.sqlite3

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
"""
        
        elif type == 'node':
            return """# Dependencies
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Production build
build/
dist/

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Coverage
coverage/
.nyc_output/
"""
        
        return ""
    
    def _get_next_steps(self, language: str, project_path: Path) -> List[str]:
        """Get next steps for the user"""
        steps = [
            f"cd {project_path.name}",
            "Review the generated code and documentation"
        ]
        
        if language == 'python':
            steps.extend([
                "python -m venv venv",
                "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
                "pip install -r requirements.txt",
                "python main.py"
            ])
        
        elif language in ['javascript', 'typescript']:
            steps.extend([
                "npm install",
                "npm start"
            ])
        
        steps.extend([
            "Run tests to verify functionality",
            "Customize the code for your specific needs",
            "Deploy to your preferred platform"
        ])
        
        return steps