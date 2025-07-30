"""
AI-Powered Code Modification System
Handles intelligent code refactoring, generation, and modification
"""
import os
import ast
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import difflib
from app.logger import logger
from app.llm import LLMClient

class AICodeModifier:
    """AI-powered code modification and refactoring system"""
    
    def __init__(self, llm_client: LLMClient, workspace_root: str = "workspace"):
        self.llm_client = llm_client
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(exist_ok=True)
        
        # Language-specific configurations
        self.language_configs = {
            'python': {
                'extensions': ['.py'],
                'comment_style': '#',
                'indent': '    ',
                'imports_section': True,
                'ast_parser': ast.parse
            },
            'javascript': {
                'extensions': ['.js', '.jsx'],
                'comment_style': '//',
                'indent': '  ',
                'imports_section': True,
                'ast_parser': None
            },
            'typescript': {
                'extensions': ['.ts', '.tsx'],
                'comment_style': '//',
                'indent': '  ',
                'imports_section': True,
                'ast_parser': None
            },
            'java': {
                'extensions': ['.java'],
                'comment_style': '//',
                'indent': '    ',
                'imports_section': True,
                'ast_parser': None
            },
            'cpp': {
                'extensions': ['.cpp', '.c', '.h', '.hpp'],
                'comment_style': '//',
                'indent': '    ',
                'imports_section': True,
                'ast_parser': None
            }
        }
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        for lang, config in self.language_configs.items():
            if file_ext in config['extensions']:
                return lang
        
        return None
    
    def analyze_code_structure(self, code: str, language: str) -> Dict:
        """Analyze code structure and extract metadata"""
        try:
            analysis = {
                'language': language,
                'lines': len(code.split('\n')),
                'functions': [],
                'classes': [],
                'imports': [],
                'comments': [],
                'todos': [],
                'complexity_score': 0
            }
            
            lines = code.split('\n')
            
            if language == 'python':
                try:
                    tree = ast.parse(code)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            analysis['functions'].append({
                                'name': node.name,
                                'line': node.lineno,
                                'args': [arg.arg for arg in node.args.args],
                                'docstring': ast.get_docstring(node)
                            })
                        elif isinstance(node, ast.ClassDef):
                            analysis['classes'].append({
                                'name': node.name,
                                'line': node.lineno,
                                'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                                'docstring': ast.get_docstring(node)
                            })
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                analysis['imports'].append({
                                    'type': 'import',
                                    'module': alias.name,
                                    'alias': alias.asname,
                                    'line': node.lineno
                                })
                        elif isinstance(node, ast.ImportFrom):
                            analysis['imports'].append({
                                'type': 'from_import',
                                'module': node.module,
                                'names': [alias.name for alias in node.names],
                                'line': node.lineno
                            })
                
                except SyntaxError as e:
                    analysis['syntax_error'] = str(e)
            
            # General analysis for all languages
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Find comments
                comment_char = self.language_configs[language]['comment_style']
                if stripped.startswith(comment_char):
                    analysis['comments'].append({'line': i, 'content': stripped})
                
                # Find TODOs
                if 'todo' in stripped.lower() or 'fixme' in stripped.lower():
                    analysis['todos'].append({'line': i, 'content': stripped})
            
            # Calculate basic complexity score
            analysis['complexity_score'] = len(analysis['functions']) + len(analysis['classes']) * 2
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing code structure: {e}")
            return {'error': str(e)}
    
    def generate_code(self, prompt: str, language: str, context: Optional[str] = None) -> Dict:
        """Generate code based on natural language prompt"""
        try:
            system_prompt = f"""You are an expert {language} programmer. Generate clean, efficient, and well-documented code based on the user's requirements.

Guidelines:
1. Write production-ready code with proper error handling
2. Include appropriate comments and docstrings
3. Follow {language} best practices and conventions
4. Make the code modular and maintainable
5. Include type hints where applicable

Language: {language}
"""

            if context:
                system_prompt += f"\nContext/Existing code:\n{context}\n"

            user_prompt = f"Generate {language} code for: {prompt}"

            response = self.llm_client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            generated_code = response.get('content', '')
            
            # Extract code from markdown blocks if present
            code_blocks = re.findall(r'```(?:' + language + r')?\n(.*?)\n```', generated_code, re.DOTALL)
            if code_blocks:
                generated_code = code_blocks[0]
            
            # Analyze the generated code
            analysis = self.analyze_code_structure(generated_code, language)
            
            return {
                'success': True,
                'code': generated_code,
                'analysis': analysis,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {'success': False, 'error': str(e)}
    
    def refactor_code(self, code: str, refactor_type: str, language: str, 
                     specific_instructions: Optional[str] = None) -> Dict:
        """Refactor existing code with specific improvements"""
        try:
            refactor_prompts = {
                'optimize': "Optimize this code for better performance and memory usage",
                'clean': "Clean up this code by improving readability and removing redundancy",
                'modernize': "Modernize this code to use current best practices and language features",
                'add_docs': "Add comprehensive documentation and docstrings to this code",
                'add_tests': "Add unit tests for this code",
                'extract_functions': "Refactor this code by extracting reusable functions",
                'improve_error_handling': "Improve error handling and add proper exception management",
                'add_logging': "Add appropriate logging to this code",
                'make_async': "Convert this code to use async/await patterns where beneficial"
            }
            
            base_prompt = refactor_prompts.get(refactor_type, refactor_type)
            
            system_prompt = f"""You are an expert {language} programmer specializing in code refactoring. 
Your task is to improve the given code while maintaining its functionality.

Guidelines:
1. Preserve the original functionality
2. Improve code quality, readability, and maintainability
3. Follow {language} best practices
4. Add comments explaining significant changes
5. Ensure the refactored code is production-ready

Language: {language}
Refactoring goal: {base_prompt}
"""

            if specific_instructions:
                system_prompt += f"\nSpecific instructions: {specific_instructions}"

            user_prompt = f"Refactor this {language} code:\n\n```{language}\n{code}\n```"

            response = self.llm_client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            refactored_code = response.get('content', '')
            
            # Extract code from markdown blocks
            code_blocks = re.findall(r'```(?:' + language + r')?\n(.*?)\n```', refactored_code, re.DOTALL)
            if code_blocks:
                refactored_code = code_blocks[0]
            
            # Generate diff
            diff = list(difflib.unified_diff(
                code.splitlines(keepends=True),
                refactored_code.splitlines(keepends=True),
                fromfile='original',
                tofile='refactored',
                lineterm=''
            ))
            
            # Analyze both versions
            original_analysis = self.analyze_code_structure(code, language)
            refactored_analysis = self.analyze_code_structure(refactored_code, language)
            
            return {
                'success': True,
                'original_code': code,
                'refactored_code': refactored_code,
                'diff': ''.join(diff),
                'original_analysis': original_analysis,
                'refactored_analysis': refactored_analysis,
                'refactor_type': refactor_type,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Error refactoring code: {e}")
            return {'success': False, 'error': str(e)}
    
    def explain_code(self, code: str, language: str) -> Dict:
        """Generate detailed explanation of code functionality"""
        try:
            system_prompt = f"""You are an expert {language} programmer and teacher. 
Provide a comprehensive explanation of the given code, including:

1. Overall purpose and functionality
2. Key components and their roles
3. Algorithm or logic flow
4. Important design patterns or techniques used
5. Potential improvements or considerations
6. Line-by-line breakdown for complex sections

Be clear and educational in your explanation."""

            user_prompt = f"Explain this {language} code:\n\n```{language}\n{code}\n```"

            response = self.llm_client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            explanation = response.get('content', '')
            
            # Analyze the code structure
            analysis = self.analyze_code_structure(code, language)
            
            return {
                'success': True,
                'explanation': explanation,
                'analysis': analysis,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Error explaining code: {e}")
            return {'success': False, 'error': str(e)}
    
    def fix_code_issues(self, code: str, language: str, error_message: Optional[str] = None) -> Dict:
        """Fix syntax errors and common issues in code"""
        try:
            system_prompt = f"""You are an expert {language} programmer specializing in debugging and fixing code issues.
Your task is to identify and fix problems in the given code.

Guidelines:
1. Fix syntax errors and runtime issues
2. Improve code quality and best practices
3. Maintain original functionality
4. Add comments explaining fixes
5. Ensure the code follows {language} conventions

Language: {language}
"""

            user_prompt = f"Fix the issues in this {language} code:\n\n```{language}\n{code}\n```"
            
            if error_message:
                user_prompt += f"\n\nError message: {error_message}"

            response = self.llm_client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            fixed_code = response.get('content', '')
            
            # Extract code from markdown blocks
            code_blocks = re.findall(r'```(?:' + language + r')?\n(.*?)\n```', fixed_code, re.DOTALL)
            if code_blocks:
                fixed_code = code_blocks[0]
            
            # Generate diff
            diff = list(difflib.unified_diff(
                code.splitlines(keepends=True),
                fixed_code.splitlines(keepends=True),
                fromfile='original',
                tofile='fixed',
                lineterm=''
            ))
            
            # Analyze both versions
            original_analysis = self.analyze_code_structure(code, language)
            fixed_analysis = self.analyze_code_structure(fixed_code, language)
            
            return {
                'success': True,
                'original_code': code,
                'fixed_code': fixed_code,
                'diff': ''.join(diff),
                'original_analysis': original_analysis,
                'fixed_analysis': fixed_analysis,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Error fixing code: {e}")
            return {'success': False, 'error': str(e)}
    
    def convert_code(self, code: str, source_language: str, target_language: str) -> Dict:
        """Convert code from one language to another"""
        try:
            system_prompt = f"""You are an expert programmer fluent in multiple programming languages.
Convert the given {source_language} code to {target_language} while maintaining functionality.

Guidelines:
1. Preserve the original logic and functionality
2. Use idiomatic {target_language} patterns and conventions
3. Adapt data structures and syntax appropriately
4. Include necessary imports and dependencies
5. Add comments explaining language-specific adaptations

Source Language: {source_language}
Target Language: {target_language}
"""

            user_prompt = f"Convert this {source_language} code to {target_language}:\n\n```{source_language}\n{code}\n```"

            response = self.llm_client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            converted_code = response.get('content', '')
            
            # Extract code from markdown blocks
            code_blocks = re.findall(r'```(?:' + target_language + r')?\n(.*?)\n```', converted_code, re.DOTALL)
            if code_blocks:
                converted_code = code_blocks[0]
            
            # Analyze both versions
            source_analysis = self.analyze_code_structure(code, source_language)
            target_analysis = self.analyze_code_structure(converted_code, target_language)
            
            return {
                'success': True,
                'source_code': code,
                'converted_code': converted_code,
                'source_language': source_language,
                'target_language': target_language,
                'source_analysis': source_analysis,
                'target_analysis': target_analysis
            }
            
        except Exception as e:
            logger.error(f"Error converting code: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_tests(self, code: str, language: str, test_framework: Optional[str] = None) -> Dict:
        """Generate unit tests for the given code"""
        try:
            # Default test frameworks by language
            default_frameworks = {
                'python': 'pytest',
                'javascript': 'jest',
                'typescript': 'jest',
                'java': 'junit',
                'cpp': 'gtest'
            }
            
            if not test_framework:
                test_framework = default_frameworks.get(language, 'standard')
            
            system_prompt = f"""You are an expert {language} programmer specializing in test-driven development.
Generate comprehensive unit tests for the given code using {test_framework}.

Guidelines:
1. Test all public functions and methods
2. Include edge cases and error conditions
3. Use appropriate assertions and test patterns
4. Follow {test_framework} best practices
5. Include setup and teardown when necessary
6. Add descriptive test names and comments

Language: {language}
Test Framework: {test_framework}
"""

            user_prompt = f"Generate unit tests for this {language} code:\n\n```{language}\n{code}\n```"

            response = self.llm_client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            test_code = response.get('content', '')
            
            # Extract code from markdown blocks
            code_blocks = re.findall(r'```(?:' + language + r')?\n(.*?)\n```', test_code, re.DOTALL)
            if code_blocks:
                test_code = code_blocks[0]
            
            # Analyze the original code and generated tests
            code_analysis = self.analyze_code_structure(code, language)
            test_analysis = self.analyze_code_structure(test_code, language)
            
            return {
                'success': True,
                'original_code': code,
                'test_code': test_code,
                'test_framework': test_framework,
                'language': language,
                'code_analysis': code_analysis,
                'test_analysis': test_analysis
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {'success': False, 'error': str(e)}
    
    def modify_file(self, file_path: str, modification_type: str, 
                   instructions: str, backup: bool = True) -> Dict:
        """Modify a code file with AI assistance"""
        try:
            # Resolve file path
            if not Path(file_path).is_absolute():
                file_path = self.workspace_root / file_path
            else:
                file_path = Path(file_path)
            
            # Security check
            if not str(file_path).startswith(str(self.workspace_root)):
                return {'success': False, 'error': 'File path outside workspace not allowed'}
            
            if not file_path.exists():
                return {'success': False, 'error': 'File does not exist'}
            
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # Create backup if requested
            if backup:
                backup_path = file_path.with_suffix(f'.bak.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_code)
            
            # Detect language
            language = self.detect_language(str(file_path))
            if not language:
                return {'success': False, 'error': 'Unsupported file type'}
            
            # Apply modification based on type
            if modification_type == 'refactor':
                result = self.refactor_code(original_code, instructions, language)
            elif modification_type == 'fix':
                result = self.fix_code_issues(original_code, language, instructions)
            elif modification_type == 'enhance':
                result = self.refactor_code(original_code, 'enhance', language, instructions)
            else:
                return {'success': False, 'error': f'Unknown modification type: {modification_type}'}
            
            if not result.get('success'):
                return result
            
            # Get modified code
            if modification_type == 'refactor' or modification_type == 'enhance':
                modified_code = result['refactored_code']
            elif modification_type == 'fix':
                modified_code = result['fixed_code']
            
            # Write modified file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            
            return {
                'success': True,
                'message': f'File modified successfully: {file_path.name}',
                'file_path': str(file_path),
                'backup_path': str(backup_path) if backup else None,
                'modification_type': modification_type,
                'original_size': len(original_code),
                'modified_size': len(modified_code),
                'diff': result.get('diff', ''),
                'language': language
            }
            
        except Exception as e:
            logger.error(f"Error modifying file: {e}")
            return {'success': False, 'error': str(e)}