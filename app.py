
# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import mimetypes
import os
import time
from pathlib import Path
import asyncio
import queue
from app.agent.manus import Manus
from app.logger import logger, log_queue
from app.config import config as app_config
from app.git_manager import GitManager
from app.file_manager import AdvancedFileManager
from app.ai_code_modifier import AICodeModifier
from app.project_templates import ProjectTemplateManager
from app.llm import LLMClient
import threading
import toml
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import uuid
from werkzeug.utils import secure_filename
import shutil
from app.tool.manus_agent import ManusAgent
from app.tool.advanced_code_generator import AdvancedCodeGenerator

app = Flask(__name__)
app.config['WORKSPACE'] = 'workspace'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHAT_HISTORY_FILE'] = 'chat_history.json'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['WORKSPACE'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to track running tasks
running_tasks = {}

# Load configuration
config = toml.load('config/config.toml')

# Advanced API Key Management System
class AdvancedAPIKeyManager:
    def __init__(self, api_keys_config):
        self.api_keys = []
        self.usage_stats = {}
        self.disabled_keys = {}  # {key: disabled_until_timestamp}
        self.failure_counts = {}  # {key: consecutive_failures}
        self.last_used = {}  # {key: last_used_timestamp}
        
        # Initialize API keys from config
        for key_config in api_keys_config:
            self.api_keys.append({
                'api_key': key_config['api_key'],
                'name': key_config.get('name', f"Key_{key_config['api_key'][:8]}"),
                'max_requests_per_minute': key_config.get('max_requests_per_minute', 5),
                'max_requests_per_hour': key_config.get('max_requests_per_hour', 100),
                'max_requests_per_day': key_config.get('max_requests_per_day', 100),
                'priority': key_config.get('priority', 1),
                'enabled': key_config.get('enabled', True)
            })
            
            # Initialize stats for each key
            key = key_config['api_key']
            self.usage_stats[key] = {
                'requests_this_minute': [],
                'requests_this_hour': [],
                'requests_this_day': [],
                'total_requests': 0
            }
            self.failure_counts[key] = 0
            self.last_used[key] = None
        
        logger.info(f"Initialized advanced API key manager with {len(self.api_keys)} keys")
    
    def _clean_old_usage_data(self, api_key: str):
        """Clean old usage data for accurate rate limiting"""
        current_time = time.time()
        stats = self.usage_stats[api_key]
        
        # Clean minute data (older than 60 seconds)
        stats['requests_this_minute'] = [
            t for t in stats['requests_this_minute'] 
            if current_time - t < 60
        ]
        
        # Clean hour data (older than 3600 seconds)
        stats['requests_this_hour'] = [
            t for t in stats['requests_this_hour'] 
            if current_time - t < 3600
        ]
        
        # Clean day data (older than 86400 seconds)
        stats['requests_this_day'] = [
            t for t in stats['requests_this_day'] 
            if current_time - t < 86400
        ]
    
    def _is_key_available(self, key_config: dict) -> bool:
        """Check if an API key is available for use"""
        api_key = key_config['api_key']
        current_time = time.time()
        
        # Check if key is disabled
        if not key_config['enabled']:
            return False
        
        # Check if key is in cooldown period (24 hours after rate limit)
        if api_key in self.disabled_keys:
            if current_time < self.disabled_keys[api_key]:
                logger.debug(f"Key {key_config['name']} still in cooldown")
                return False
            else:
                # Cooldown expired, remove from disabled list
                del self.disabled_keys[api_key]
                self.failure_counts[api_key] = 0  # Reset failure count
                logger.info(f"Key {key_config['name']} cooldown expired, re-enabling")
        
        # Clean old usage data
        self._clean_old_usage_data(api_key)
        
        # Check rate limits
        stats = self.usage_stats[api_key]
        
        if len(stats['requests_this_minute']) >= key_config['max_requests_per_minute']:
            logger.debug(f"Key {key_config['name']} hit minute limit")
            return False
        
        if len(stats['requests_this_hour']) >= key_config['max_requests_per_hour']:
            logger.debug(f"Key {key_config['name']} hit hour limit")
            return False
        
        if len(stats['requests_this_day']) >= key_config['max_requests_per_day']:
            logger.debug(f"Key {key_config['name']} hit daily limit")
            # Disable key for 24 hours
            self._disable_key_for_rate_limit(api_key, key_config['name'])
            return False
        
        return True
    
    def _disable_key_for_rate_limit(self, api_key: str, key_name: str):
        """Disable API key for 24 hours due to rate limit"""
        disable_until = time.time() + 24 * 60 * 60  # 24 hours
        self.disabled_keys[api_key] = disable_until
        
        logger.warning(f"API key {key_name} disabled for 24 hours due to rate limit")
    
    def _calculate_key_score(self, key_config: dict) -> float:
        """Calculate a score for key selection (higher is better)"""
        api_key = key_config['api_key']
        current_time = time.time()
        
        # Base score from priority (lower priority number = higher score)
        priority_score = 10.0 / max(key_config['priority'], 1)
        
        # Usage-based score (less recent usage = higher score)
        stats = self.usage_stats[api_key]
        minute_usage = len(stats['requests_this_minute'])
        hour_usage = len(stats['requests_this_hour'])
        day_usage = len(stats['requests_this_day'])
        
        # Calculate remaining capacity
        minute_capacity = 1.0 - (minute_usage / key_config['max_requests_per_minute'])
        hour_capacity = 1.0 - (hour_usage / key_config['max_requests_per_hour'])
        day_capacity = 1.0 - (day_usage / key_config['max_requests_per_day'])
        
        capacity_score = (minute_capacity + hour_capacity + day_capacity) / 3
        
        # Failure-based score (fewer failures = higher score)
        failure_score = 1.0 / (self.failure_counts[api_key] + 1)
        
        # Time since last use (longer = slightly higher score)
        time_score = 1.0
        if self.last_used[api_key]:
            time_since_use = current_time - self.last_used[api_key]
            time_score = min(1.0 + (time_since_use / 3600), 2.0)  # Max 2x after 1 hour
        
        # Combine all factors
        final_score = priority_score * capacity_score * failure_score * time_score
        return max(final_score, 0.1)  # Minimum score
    
    def get_available_api_key(self, use_random: bool = True) -> Optional[Tuple[str, dict]]:
        """Get an available API key with advanced selection logic"""
        available_keys = []
        
        # Find all available keys
        for key_config in self.api_keys:
            if self._is_key_available(key_config):
                available_keys.append(key_config)
        
        if not available_keys:
            logger.warning("No API keys available")
            return None
        
        if use_random and len(available_keys) > 1:
            # Advanced weighted random selection
            weights = []
            for key_config in available_keys:
                score = self._calculate_key_score(key_config)
                weights.append(score)
            
            # Weighted random choice
            selected_key = random.choices(available_keys, weights=weights)[0]
            logger.info(f"Randomly selected API key: {selected_key['name']} (weighted selection)")
        else:
            # Priority-based selection with health metrics
            available_keys.sort(key=lambda k: (
                k['priority'],
                -self._calculate_key_score(k),
                self.failure_counts[k['api_key']],
                k['api_key']  # Deterministic tie-breaker
            ))
            selected_key = available_keys[0]
            logger.info(f"Priority selected API key: {selected_key['name']}")
        
        return selected_key['api_key'], selected_key
    
    def record_successful_request(self, api_key: str):
        """Record a successful API request"""
        current_time = time.time()
        stats = self.usage_stats[api_key]
        
        # Add timestamps
        stats['requests_this_minute'].append(current_time)
        stats['requests_this_hour'].append(current_time)
        stats['requests_this_day'].append(current_time)
        stats['total_requests'] += 1
        
        # Update last used time
        self.last_used[api_key] = current_time
        
        # Reset failure count on success
        self.failure_counts[api_key] = 0
        
        logger.info(f"Recorded successful request for API key")
    
    def record_rate_limit_error(self, api_key: str, key_name: str):
        """Record a rate limit error and disable the key"""
        self._disable_key_for_rate_limit(api_key, key_name)
        self.failure_counts[api_key] += 1
        logger.warning(f"Rate limit error recorded for {key_name}")
    
    def record_failure(self, api_key: str, key_name: str, error_type: str = "unknown"):
        """Record a failure for an API key"""
        self.failure_counts[api_key] += 1
        logger.warning(f"Failure recorded for {key_name}: {error_type} (consecutive: {self.failure_counts[api_key]})")
        
        # If too many consecutive failures, disable temporarily
        if self.failure_counts[api_key] >= 5:
            # Exponential backoff: 5 minutes * 2^(failures-5)
            backoff_minutes = 5 * (2 ** (self.failure_counts[api_key] - 5))
            backoff_minutes = min(backoff_minutes, 240)  # Cap at 4 hours
            
            disable_until = time.time() + (backoff_minutes * 60)
            self.disabled_keys[api_key] = disable_until
            logger.warning(f"Temporarily disabled {key_name} for {backoff_minutes} minutes due to failures")
    
    def get_keys_status(self) -> List[Dict]:
        """Get detailed status of all API keys"""
        status_list = []
        current_time = time.time()
        
        for key_config in self.api_keys:
            api_key = key_config['api_key']
            self._clean_old_usage_data(api_key)
            
            stats = self.usage_stats[api_key]
            is_available = self._is_key_available(key_config)
            
            status = {
                'name': key_config['name'],
                'enabled': key_config['enabled'],
                'available': is_available,
                'usage': {
                    'requests_this_minute': len(stats['requests_this_minute']),
                    'requests_this_hour': len(stats['requests_this_hour']),
                    'requests_this_day': len(stats['requests_this_day']),
                    'total_requests': stats['total_requests']
                },
                'limits': {
                    'max_per_minute': key_config['max_requests_per_minute'],
                    'max_per_hour': key_config['max_requests_per_hour'],
                    'max_per_day': key_config['max_requests_per_day']
                },
                'failures': self.failure_counts[api_key],
                'last_used': datetime.fromtimestamp(self.last_used[api_key]).isoformat() if self.last_used[api_key] else "Never"
            }
            
            # Add cooldown info if applicable
            if api_key in self.disabled_keys:
                remaining_time = int(self.disabled_keys[api_key] - current_time)
                if remaining_time > 0:
                    status['cooldown_remaining_seconds'] = remaining_time
                    status['cooldown_remaining_readable'] = f"{remaining_time // 3600}h {(remaining_time % 3600) // 60}m"
            
            status_list.append(status)
        
        return status_list

# Initialize the advanced API key manager
api_key_manager = AdvancedAPIKeyManager(config['llm']['api_keys'])

# Initialize enhanced components
git_manager = GitManager(app.config['WORKSPACE'])
file_manager = AdvancedFileManager(app.config['WORKSPACE'], app.config['UPLOAD_FOLDER'])
project_template_manager = ProjectTemplateManager(app.config['WORKSPACE'])

# Initialize enhanced Manus agent
manus_agent = ManusAgent(api_key_manager)

# Get AI code modifier instance
def get_ai_code_modifier():
    """Get AI code modifier instance with available LLM client"""
    try:
        available_key = api_key_manager.get_available_api_key()
        if available_key:
            llm_client = LLMClient(
                api_key=available_key['api_key'],
                provider=available_key['provider'],
                model=available_key.get('model', 'gpt-3.5-turbo')
            )
            return AICodeModifier(llm_client)
        return None
    except Exception as e:
        logger.error(f"Error getting AI code modifier: {e}")
        return None

# Enhanced main function with Manus agent integration
def main(query: str, flow_id: str = None) -> Dict[str, Any]:
    """Enhanced main function with comprehensive AI assistance"""
    try:
        logger.info(f"Processing query with Manus Agent: {query}")
        
        # Use Manus agent for all AI-powered tasks
        result = manus_agent.execute(request=query)
        
        if result['success']:
            logger.info(f"Manus Agent completed successfully: {result.get('message', '')}")
            return {
                'success': True,
                'result': result,
                'agent': 'manus',
                'flow_id': flow_id,
                'timestamp': datetime.now().isoformat()
            }
        else:
            logger.warning(f"Manus Agent failed: {result.get('error', 'Unknown error')}")
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'agent': 'manus',
                'flow_id': flow_id,
                'timestamp': datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return {
            'success': False,
            'error': str(e),
            'flow_id': flow_id,
            'timestamp': datetime.now().isoformat()
        }

# Enhanced API endpoint for Manus agent
@app.route('/api/manus', methods=['POST'])
def api_manus():
    """Enhanced API endpoint for Manus agent with comprehensive features"""
    try:
        data = request.get_json()
        
        if not data or 'request' not in data:
            return jsonify({
                'success': False,
                'error': 'Request is required',
                'example': {
                    'request': 'Create a Flask web application with user authentication',
                    'task_type': 'project_generation',  # optional
                    'options': {}  # optional
                }
            }), 400
        
        # Execute with Manus agent
        result = manus_agent.execute(
            request=data['request'],
            task_type=data.get('task_type', 'auto_detect'),
            **data.get('options', {})
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in Manus API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint for advanced code generation
@app.route('/api/manus/generate-project', methods=['POST'])
def api_generate_project():
    """API endpoint for advanced project generation"""
    try:
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({
                'success': False,
                'error': 'Project description is required'
            }), 400
        
        # Use Manus agent for project generation
        result = manus_agent.execute(
            request=data['description'],
            task_type='project_generation',
            **data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in project generation API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint for code testing
@app.route('/api/manus/test-code', methods=['POST'])
def api_test_code():
    """API endpoint for code testing and validation"""
    try:
        data = request.get_json()
        
        # Use Manus agent for testing
        result = manus_agent.execute(
            request=data.get('request', 'Generate comprehensive tests for my code'),
            task_type='testing',
            **data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in testing API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint for bug fixing
@app.route('/api/manus/fix-bugs', methods=['POST'])
def api_fix_bugs():
    """API endpoint for bug fixing and code repair"""
    try:
        data = request.get_json()
        
        # Use Manus agent for bug fixing
        result = manus_agent.execute(
            request=data.get('request', 'Fix bugs and issues in my code'),
            task_type='bug_fixing',
            **data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in bug fixing API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint for code analysis
@app.route('/api/manus/analyze-code', methods=['POST'])
def api_analyze_code():
    """API endpoint for code analysis and review"""
    try:
        data = request.get_json()
        
        # Use Manus agent for code analysis
        result = manus_agent.execute(
            request=data.get('request', 'Analyze my code quality and structure'),
            task_type='code_analysis',
            **data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in code analysis API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Enhanced API endpoint for file operations via Manus
@app.route('/api/manus/file-operation', methods=['POST'])
def api_manus_file_operation():
    """API endpoint for file operations via Manus agent"""
    try:
        data = request.get_json()
        
        if not data or 'request' not in data:
            return jsonify({
                'success': False,
                'error': 'File operation request is required'
            }), 400
        
        # Use Manus agent for file operations
        result = manus_agent.execute(
            request=data['request'],
            task_type='file_management',
            **data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in file operation API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Enhanced API endpoint for Git operations via Manus
@app.route('/api/manus/git-operation', methods=['POST'])
def api_manus_git_operation():
    """API endpoint for Git operations via Manus agent"""
    try:
        data = request.get_json()
        
        if not data or 'request' not in data:
            return jsonify({
                'success': False,
                'error': 'Git operation request is required'
            }), 400
        
        # Use Manus agent for Git operations
        result = manus_agent.execute(
            request=data['request'],
            task_type='git_operations',
            **data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in Git operation API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint for Manus agent conversation
@app.route('/api/manus/chat', methods=['POST'])
def api_manus_chat():
    """API endpoint for conversational AI assistance"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Message is required for chat'
            }), 400
        
        # Use Manus agent for conversation
        result = manus_agent.execute(
            request=data['message'],
            task_type='conversation'
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint to get Manus agent status and capabilities
@app.route('/api/manus/status', methods=['GET'])
def api_manus_status():
    """Get Manus agent status and capabilities"""
    try:
        # Check if AI features are available
        ai_available = manus_agent.ai_modifier is not None
        code_gen_available = manus_agent.code_generator is not None
        
        status = {
            'success': True,
            'agent_status': 'active',
            'capabilities': {
                'project_generation': code_gen_available,
                'code_testing': ai_available,
                'bug_fixing': ai_available,
                'code_analysis': ai_available,
                'file_management': True,
                'git_operations': True,
                'conversation': True
            },
            'task_history_count': len(manus_agent.get_task_history()),
            'conversation_context_count': len(manus_agent.get_conversation_context()),
            'workspace_root': str(manus_agent.workspace_root),
            'available_languages': ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'csharp', 'php', 'ruby', 'go'],
            'supported_frameworks': ['flask', 'fastapi', 'django', 'react', 'vue', 'express', 'nodejs']
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting Manus status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint to get task history
@app.route('/api/manus/history', methods=['GET'])
def api_manus_history():
    """Get Manus agent task history"""
    try:
        history = manus_agent.get_task_history()
        return jsonify({
            'success': True,
            'history': history,
            'total_tasks': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting task history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint to get conversation context
@app.route('/api/manus/context', methods=['GET'])
def api_manus_context():
    """Get Manus agent conversation context"""
    try:
        context = manus_agent.get_conversation_context()
        return jsonify({
            'success': True,
            'context': context,
            'context_length': len(context)
        })
        
    except Exception as e:
        logger.error(f"Error getting conversation context: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Enhanced file upload with AI processing
@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Enhanced file upload with AI processing capabilities"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join('workspace', 'uploads', filename)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        
        file.save(upload_path)
        
        # Check if AI processing is requested
        ai_process = request.form.get('ai_process', 'false').lower() == 'true'
        
        result = {
            'success': True,
            'filename': filename,
            'path': upload_path,
            'size': os.path.getsize(upload_path)
        }
        
        if ai_process:
            # Use Manus agent to analyze the uploaded file
            analysis_request = f"Analyze the uploaded file: {filename}"
            analysis_result = manus_agent.execute(
                request=analysis_request,
                task_type='code_analysis',
                target_path=upload_path
            )
            
            result['ai_analysis'] = analysis_result
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Thread wrapper
def run_async_task(message, task_id=None):
    """Run async task in new thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(message, task_id))
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
    finally:
        loop.close()

@app.route('/api/chat-stream', methods=['POST'])
def chat_stream():
    """Enhanced streaming chat interface with stop functionality"""
    # Clear log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # Get request data
    prompt_data = request.get_json()
    message = prompt_data["message"]
    task_id = prompt_data.get("task_id", str(uuid.uuid4()))
    uploaded_files = prompt_data.get("uploaded_files", [])
    
    logger.info(f"Received request: {message}")
    
    # Process uploaded files if any
    file_context = ""
    if uploaded_files:
        file_context = "\n\nUploaded files context:\n"
        for file_info in uploaded_files:
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_info['filename'])
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:2000]  # Limit content size
                    file_context += f"\n--- {file_info['original_name']} ---\n{content}\n"
            except Exception as e:
                logger.error(f"Error reading file {file_info['filename']}: {e}")
    
    full_message = message + file_context
    
    # Initialize task tracking
    running_tasks[task_id] = {
        'stop_flag': False,
        'start_time': time.time()
    }

    # Start async task thread
    task_thread = threading.Thread(
        target=run_async_task,
        args=(full_message, task_id)
    )
    task_thread.start()

    # Streaming generator
    def generate():
        start_time = time.time()
        full_response = ""

        while task_thread.is_alive() or not log_queue.empty():
            # Check for stop signal
            if running_tasks.get(task_id, {}).get('stop_flag', False):
                yield "Task stopped by user.\n"
                break
                
            # Timeout check
            if time.time() - start_time > PROCESS_TIMEOUT:
                yield """0303030"""
                break
            
            new_content = ""
            try:
                new_content = log_queue.get(timeout=0.1)
            except queue.Empty:
                pass

            if new_content:
                full_response += new_content
                yield new_content

            # Pause when no new content
            if not new_content:
                time.sleep(FILE_CHECK_INTERVAL)

        # Save chat history
        chat_data = {
            'id': task_id,
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'agent_response': full_response,
            'agent_type': 'manus',
            'uploaded_files': uploaded_files
        }
        save_chat_history(chat_data)
        
        # Clean up task tracking
        if task_id in running_tasks:
            del running_tasks[task_id]

        # Final confirmation
        yield """0303030"""

    return Response(generate(), mimetype="text/plain")

# Run flow async task
async def run_flow_task(prompt, task_id=None):
    """Enhanced run_flow function with advanced API key rotation and stop functionality"""
    from app.agent.data_analysis import DataAnalysis
    from app.flow.flow_factory import FlowFactory, FlowType
    
    max_retries = len(api_key_manager.api_keys)
    retry_count = 0
    
    while retry_count < max_retries:
        # Get available API key
        result = api_key_manager.get_available_api_key(use_random=True)
        if not result:
            logger.error("No API keys available for request")
            
            # Wait for next available key
            max_wait_time = 300  # 5 minutes
            wait_time = 5
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                logger.info(f"Waiting {wait_time}s for API key availability...")
                await asyncio.sleep(wait_time)
                
                result = api_key_manager.get_available_api_key(use_random=True)
                if result:
                    break
                    
                wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s
            
            if not result:
                raise Exception("No API keys became available within timeout period")
        
        api_key, key_config = result
        key_name = key_config['name']
        
        try:
            logger.info(f"Using API key: {key_name}")
            
            # Create agents with advanced API key manager
            agents = {
                "manus": await Manus.create(
                    api_key_manager=api_key_manager,
                    api_key=api_key
                ),
            }
            
            if app_config.run_flow_config.use_data_analysis_agent:
                agents["data_analysis"] = DataAnalysis()
            
            # Create and execute flow
            flow = FlowFactory.create_flow(
                flow_type=FlowType.PLANNING,
                agents=agents,
            )
            
            logger.warning("Processing your request with flow...")
            
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    flow.execute(prompt),
                    timeout=3600,  # 60 minute timeout for the entire execution
                )
                elapsed_time = time.time() - start_time
                logger.info(f"Request processed in {elapsed_time:.2f} seconds")
                logger.info(result)
                
                # Record successful request
                api_key_manager.record_successful_request(api_key)
                logger.info(f"Flow task completed successfully with key: {key_name}")
                break
                
            except asyncio.TimeoutError:
                logger.error("Request processing timed out after 1 hour")
                logger.info("Operation terminated due to timeout. Please try a simpler request.")
                break
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle different types of errors
            if any(keyword in error_str for keyword in ["rate limit", "quota", "too many requests"]):
                logger.warning(f"Rate limit error with key {key_name}: {e}")
                api_key_manager.record_rate_limit_error(api_key, key_name)
            elif any(keyword in error_str for keyword in ["authentication", "invalid api key", "unauthorized"]):
                logger.error(f"Authentication error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "auth_error")
            elif any(keyword in error_str for keyword in ["timeout", "connection"]):
                logger.warning(f"Connection error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "connection_error")
            else:
                logger.error(f"Unexpected error with key {key_name}: {e}")
                api_key_manager.record_failure(api_key, key_name, "unknown_error")
            
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("All API keys exhausted, flow task failed")
                raise Exception(f"Flow task failed after trying all available API keys. Last error: {e}")
            
            logger.info(f"Retrying with different API key (attempt {retry_count + 1}/{max_retries})")
            
        finally:
            if 'agents' in locals():
                for agent in agents.values():
                    if hasattr(agent, 'cleanup'):
                        await agent.cleanup()

def run_flow_async_task(message, task_id=None):
    """Run flow async task in new thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_flow_task(message, task_id))
    except Exception as e:
        logger.error(f"Flow task execution failed: {e}")
    finally:
        loop.close()

@app.route('/api/flow-stream', methods=['POST'])
def flow_stream():
    """Enhanced Flow streaming interface with stop functionality"""
    # Clear log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # Get request data
    prompt_data = request.get_json()
    message = prompt_data["message"]
    task_id = prompt_data.get("task_id", str(uuid.uuid4()))
    uploaded_files = prompt_data.get("uploaded_files", [])
    
    logger.info(f"Received Flow request: {message}")
    
    # Process uploaded files if any
    file_context = ""
    if uploaded_files:
        file_context = "\n\nUploaded files context:\n"
        for file_info in uploaded_files:
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_info['filename'])
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:2000]  # Limit content size
                    file_context += f"\n--- {file_info['original_name']} ---\n{content}\n"
            except Exception as e:
                logger.error(f"Error reading file {file_info['filename']}: {e}")
    
    full_message = message + file_context
    
    # Initialize task tracking
    running_tasks[task_id] = {
        'stop_flag': False,
        'start_time': time.time()
    }

    # Start async task thread
    task_thread = threading.Thread(
        target=run_flow_async_task,
        args=(full_message, task_id)
    )
    task_thread.start()

    # Streaming generator
    def generate():
        start_time = time.time()
        full_response = ""

        while task_thread.is_alive() or not log_queue.empty():
            # Check for stop signal
            if running_tasks.get(task_id, {}).get('stop_flag', False):
                yield "Task stopped by user.\n"
                break
                
            # Timeout check
            if time.time() - start_time > PROCESS_TIMEOUT:
                yield """0303030"""
                break
            
            new_content = ""
            try:
                new_content = log_queue.get(timeout=0.1)
            except queue.Empty:
                pass

            if new_content:
                full_response += new_content
                yield new_content

            # Pause when no new content
            if not new_content:
                time.sleep(FILE_CHECK_INTERVAL)

        # Save chat history
        chat_data = {
            'id': task_id,
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'agent_response': full_response,
            'agent_type': 'flow',
            'uploaded_files': uploaded_files
        }
        save_chat_history(chat_data)
        
        # Clean up task tracking
        if task_id in running_tasks:
            del running_tasks[task_id]

        # Final confirmation
        yield """0303030"""

    return Response(generate(), mimetype="text/plain")

# =============================================================================
# GIT MANAGEMENT API ENDPOINTS
# =============================================================================

@app.route('/api/git/repositories', methods=['GET'])
def get_repositories():
    """Get list of Git repositories in workspace"""
    try:
        repositories = git_manager.list_repositories()
        return jsonify({'success': True, 'repositories': repositories})
    except Exception as e:
        logger.error(f"Error getting repositories: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/git/clone', methods=['POST'])
def clone_repository():
    """Clone a Git repository"""
    try:
        data = request.get_json()
        repo_url = data.get('repo_url')
        target_dir = data.get('target_dir')
        branch = data.get('branch')
        
        if not repo_url:
            return jsonify({'success': False, 'error': 'Repository URL is required'}), 400
        
        result = git_manager.clone_repository(repo_url, target_dir, branch)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/git/info/<path:repo_path>', methods=['GET'])
def get_repository_info(repo_path):
    """Get repository information"""
    try:
        full_path = os.path.join(app.config['WORKSPACE'], repo_path)
        info = git_manager.get_repository_info(full_path)
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        logger.error(f"Error getting repository info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/git/branch', methods=['POST'])
def create_branch():
    """Create a new Git branch"""
    try:
        data = request.get_json()
        repo_path = data.get('repo_path')
        branch_name = data.get('branch_name')
        checkout = data.get('checkout', True)
        
        if not repo_path or not branch_name:
            return jsonify({'success': False, 'error': 'Repository path and branch name are required'}), 400
        
        full_path = os.path.join(app.config['WORKSPACE'], repo_path)
        result = git_manager.create_branch(full_path, branch_name, checkout)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating branch: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/git/checkout', methods=['POST'])
def checkout_branch():
    """Checkout a Git branch"""
    try:
        data = request.get_json()
        repo_path = data.get('repo_path')
        branch_name = data.get('branch_name')
        
        if not repo_path or not branch_name:
            return jsonify({'success': False, 'error': 'Repository path and branch name are required'}), 400
        
        full_path = os.path.join(app.config['WORKSPACE'], repo_path)
        result = git_manager.checkout_branch(full_path, branch_name)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error checking out branch: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/git/commit', methods=['POST'])
def commit_changes():
    """Commit changes to repository"""
    try:
        data = request.get_json()
        repo_path = data.get('repo_path')
        message = data.get('message')
        files = data.get('files')
        
        if not repo_path or not message:
            return jsonify({'success': False, 'error': 'Repository path and commit message are required'}), 400
        
        full_path = os.path.join(app.config['WORKSPACE'], repo_path)
        result = git_manager.commit_changes(full_path, message, files)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error committing changes: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/git/push', methods=['POST'])
def push_changes():
    """Push changes to remote repository"""
    try:
        data = request.get_json()
        repo_path = data.get('repo_path')
        remote = data.get('remote', 'origin')
        branch = data.get('branch')
        
        if not repo_path:
            return jsonify({'success': False, 'error': 'Repository path is required'}), 400
        
        full_path = os.path.join(app.config['WORKSPACE'], repo_path)
        result = git_manager.push_changes(full_path, remote, branch)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error pushing changes: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/git/pull', methods=['POST'])
def pull_changes():
    """Pull changes from remote repository"""
    try:
        data = request.get_json()
        repo_path = data.get('repo_path')
        remote = data.get('remote', 'origin')
        branch = data.get('branch')
        
        if not repo_path:
            return jsonify({'success': False, 'error': 'Repository path is required'}), 400
        
        full_path = os.path.join(app.config['WORKSPACE'], repo_path)
        result = git_manager.pull_changes(full_path, remote, branch)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error pulling changes: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================================================
# ADVANCED FILE MANAGEMENT API ENDPOINTS
# =============================================================================

@app.route('/api/files/tree', methods=['GET'])
def get_directory_tree():
    """Get directory tree structure"""
    try:
        path = request.args.get('path')
        max_depth = int(request.args.get('max_depth', 3))
        
        tree = file_manager.get_directory_tree(path, max_depth)
        return jsonify({'success': True, 'tree': tree})
        
    except Exception as e:
        logger.error(f"Error getting directory tree: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/create-directory', methods=['POST'])
def create_directory():
    """Create a new directory"""
    try:
        data = request.get_json()
        path = data.get('path')
        parents = data.get('parents', True)
        
        if not path:
            return jsonify({'success': False, 'error': 'Path is required'}), 400
        
        result = file_manager.create_directory(path, parents)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/delete', methods=['DELETE'])
def delete_item():
    """Delete a file or directory"""
    try:
        data = request.get_json()
        path = data.get('path')
        force = data.get('force', False)
        
        if not path:
            return jsonify({'success': False, 'error': 'Path is required'}), 400
        
        result = file_manager.delete_item(path, force)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error deleting item: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/copy', methods=['POST'])
def copy_item():
    """Copy a file or directory"""
    try:
        data = request.get_json()
        source = data.get('source')
        destination = data.get('destination')
        overwrite = data.get('overwrite', False)
        
        if not source or not destination:
            return jsonify({'success': False, 'error': 'Source and destination are required'}), 400
        
        result = file_manager.copy_item(source, destination, overwrite)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error copying item: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/move', methods=['POST'])
def move_item():
    """Move/rename a file or directory"""
    try:
        data = request.get_json()
        source = data.get('source')
        destination = data.get('destination')
        overwrite = data.get('overwrite', False)
        
        if not source or not destination:
            return jsonify({'success': False, 'error': 'Source and destination are required'}), 400
        
        result = file_manager.move_item(source, destination, overwrite)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error moving item: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/read', methods=['GET'])
def read_file():
    """Read file content"""
    try:
        file_path = request.args.get('path')
        encoding = request.args.get('encoding', 'utf-8')
        max_size = int(request.args.get('max_size', 10*1024*1024))
        
        if not file_path:
            return jsonify({'success': False, 'error': 'File path is required'}), 400
        
        result = file_manager.read_file(file_path, encoding, max_size)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/write', methods=['POST'])
def write_file():
    """Write content to file"""
    try:
        data = request.get_json()
        file_path = data.get('path')
        content = data.get('content')
        encoding = data.get('encoding', 'utf-8')
        create_dirs = data.get('create_dirs', True)
        
        if not file_path or content is None:
            return jsonify({'success': False, 'error': 'File path and content are required'}), 400
        
        result = file_manager.write_file(file_path, content, encoding, create_dirs)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/search', methods=['GET'])
def search_files():
    """Search for files by name or content"""
    try:
        pattern = request.args.get('pattern')
        search_path = request.args.get('path')
        case_sensitive = request.args.get('case_sensitive', 'false').lower() == 'true'
        file_content = request.args.get('file_content', 'false').lower() == 'true'
        max_results = int(request.args.get('max_results', 100))
        
        if not pattern:
            return jsonify({'success': False, 'error': 'Search pattern is required'}), 400
        
        result = file_manager.search_files(pattern, search_path, case_sensitive, file_content, max_results)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error searching files: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/info', methods=['GET'])
def get_file_info():
    """Get file information"""
    try:
        file_path = request.args.get('path')
        
        if not file_path:
            return jsonify({'success': False, 'error': 'File path is required'}), 400
        
        result = file_manager.get_file_info(file_path)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/extract', methods=['POST'])
def extract_archive():
    """Extract archive file"""
    try:
        data = request.get_json()
        archive_path = data.get('archive_path')
        destination = data.get('destination')
        
        if not archive_path:
            return jsonify({'success': False, 'error': 'Archive path is required'}), 400
        
        result = file_manager.extract_archive(archive_path, destination)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error extracting archive: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/files/create-archive', methods=['POST'])
def create_archive():
    """Create archive from files/directories"""
    try:
        data = request.get_json()
        source_path = data.get('source_path')
        archive_path = data.get('archive_path')
        archive_type = data.get('archive_type', 'zip')
        
        if not source_path or not archive_path:
            return jsonify({'success': False, 'error': 'Source and archive paths are required'}), 400
        
        result = file_manager.create_archive(source_path, archive_path, archive_type)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating archive: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================================================
# AI CODE MODIFICATION API ENDPOINTS
# =============================================================================

def get_ai_code_modifier():
    """Get AI code modifier instance, initializing if needed"""
    global ai_code_modifier
    if ai_code_modifier is None:
        # Get an available API key for the LLM client
        result = api_key_manager.get_available_api_key()
        if not result:
            raise Exception("No API keys available for AI code modification")
        
        api_key, _ = result
        llm_client = LLMClient(api_key=api_key)
        ai_code_modifier = AICodeModifier(llm_client)
    
    return ai_code_modifier

@app.route('/api/ai/generate-code', methods=['POST'])
def generate_code():
    """Generate code with AI"""
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        language = data.get('language')
        context = data.get('context')
        
        if not prompt or not language:
            return jsonify({'success': False, 'error': 'Prompt and language are required'}), 400
        
        modifier = get_ai_code_modifier()
        result = modifier.generate_code(prompt, language, context)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai/refactor-code', methods=['POST'])
def refactor_code():
    """Refactor code with AI"""
    try:
        data = request.get_json()
        code = data.get('code')
        refactor_type = data.get('refactor_type')
        language = data.get('language')
        instructions = data.get('instructions')
        
        if not code or not refactor_type or not language:
            return jsonify({'success': False, 'error': 'Code, refactor type, and language are required'}), 400
        
        modifier = get_ai_code_modifier()
        result = modifier.refactor_code(code, refactor_type, language, instructions)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error refactoring code: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai/explain-code', methods=['POST'])
def explain_code():
    """Explain code with AI"""
    try:
        data = request.get_json()
        code = data.get('code')
        language = data.get('language')
        
        if not code or not language:
            return jsonify({'success': False, 'error': 'Code and language are required'}), 400
        
        modifier = get_ai_code_modifier()
        result = modifier.explain_code(code, language)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error explaining code: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai/fix-code', methods=['POST'])
def fix_code():
    """Fix code issues with AI"""
    try:
        data = request.get_json()
        code = data.get('code')
        language = data.get('language')
        error_message = data.get('error_message')
        
        if not code or not language:
            return jsonify({'success': False, 'error': 'Code and language are required'}), 400
        
        modifier = get_ai_code_modifier()
        result = modifier.fix_code_issues(code, language, error_message)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fixing code: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai/convert-code', methods=['POST'])
def convert_code():
    """Convert code between languages with AI"""
    try:
        data = request.get_json()
        code = data.get('code')
        source_language = data.get('source_language')
        target_language = data.get('target_language')
        
        if not code or not source_language or not target_language:
            return jsonify({'success': False, 'error': 'Code, source language, and target language are required'}), 400
        
        modifier = get_ai_code_modifier()
        result = modifier.convert_code(code, source_language, target_language)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error converting code: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai/generate-tests', methods=['POST'])
def generate_tests():
    """Generate unit tests with AI"""
    try:
        data = request.get_json()
        code = data.get('code')
        language = data.get('language')
        test_framework = data.get('test_framework')
        
        if not code or not language:
            return jsonify({'success': False, 'error': 'Code and language are required'}), 400
        
        modifier = get_ai_code_modifier()
        result = modifier.generate_tests(code, language, test_framework)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating tests: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ai/modify-file', methods=['POST'])
def modify_file():
    """Modify a code file with AI assistance"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        modification_type = data.get('modification_type')
        instructions = data.get('instructions')
        backup = data.get('backup', True)
        
        if not file_path or not modification_type or not instructions:
            return jsonify({'success': False, 'error': 'File path, modification type, and instructions are required'}), 400
        
        modifier = get_ai_code_modifier()
        result = modifier.modify_file(file_path, modification_type, instructions, backup)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error modifying file: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# =============================================================================
# PROJECT TEMPLATE API ENDPOINTS
# =============================================================================

@app.route('/api/templates/list', methods=['GET'])
def list_templates():
    """List available project templates"""
    try:
        templates = project_template_manager.list_templates()
        return jsonify({'success': True, 'templates': templates})
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/templates/create-project', methods=['POST'])
def create_project():
    """Create a new project from template"""
    try:
        data = request.get_json()
        template_id = data.get('template_id')
        project_name = data.get('project_name')
        custom_vars = data.get('custom_vars', {})
        
        if not template_id or not project_name:
            return jsonify({'success': False, 'error': 'Template ID and project name are required'}), 400
        
        result = project_template_manager.create_project(template_id, project_name, custom_vars)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# WSGI entry point for deployment
application = app

if __name__ == '__main__':
    # Log initial API key status
    logger.info("=== Initial API Key Status ===")
    for status in api_key_manager.get_keys_status():
        logger.info(f"Key {status['name']}: Available={status['available']}, "
                    f"Usage={status['usage']['requests_this_day']}/{status['limits']['max_per_day']} today")
    
    app.run(host='0.0.0.0', port=3000,  debug=False)
