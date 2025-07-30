
# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session
import mimetypes
import os
import time
from pathlib import Path
import asyncio
import queue
from app.agent.manus import Manus
from app.logger import logger, log_queue
from app.config import config as app_config
import threading
import toml
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import uuid
from werkzeug.utils import secure_filename
import shutil
import zipfile
import tempfile
import subprocess
import requests
from urllib.parse import urlparse
import hashlib
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['WORKSPACE'] = 'workspace'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHAT_HISTORY_FILE'] = 'chat_history.json'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['GITHUB_CLONE_FOLDER'] = 'github_clones'
app.config['TEMP_FOLDER'] = 'temp'
app.config['USER_SESSIONS'] = {}

# Create necessary directories
os.makedirs(app.config['WORKSPACE'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GITHUB_CLONE_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# Global variable to track running tasks
running_tasks = {}

# Load configuration
config = toml.load('config/config.toml')

# Multi-user session management
def get_user_session():
    """Get or create user session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['username'] = f"User_{session['user_id'][:8]}"
        session['created_at'] = datetime.now().isoformat()
    
    user_id = session['user_id']
    if user_id not in app.config['USER_SESSIONS']:
        app.config['USER_SESSIONS'][user_id] = {
            'workspace': os.path.join(app.config['WORKSPACE'], user_id),
            'uploads': os.path.join(app.config['UPLOAD_FOLDER'], user_id),
            'github_clones': os.path.join(app.config['GITHUB_CLONE_FOLDER'], user_id),
            'temp': os.path.join(app.config['TEMP_FOLDER'], user_id),
            'chat_history': [],
            'created_at': datetime.now(),
            'last_active': datetime.now()
        }
        # Create user-specific directories
        for path in app.config['USER_SESSIONS'][user_id].values():
            if isinstance(path, str) and not path.endswith('.json'):
                os.makedirs(path, exist_ok=True)
    
    return app.config['USER_SESSIONS'][user_id]

def require_user_session(f):
    """Decorator to ensure user session exists"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        get_user_session()
        return f(*args, **kwargs)
    return decorated_function

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
        if api_key in self.disabled_keys:
            if current_time < self.disabled_keys[api_key]:
                return False
            else:
                del self.disabled_keys[api_key]
        
        # Check if key is enabled
        if not key_config.get('enabled', True):
            return False
        
        # Clean old usage data
        self._clean_old_usage_data(api_key)
        
        # Check rate limits
        stats = self.usage_stats[api_key]
        if (len(stats['requests_this_minute']) >= key_config.get('max_requests_per_minute', 5) or
            len(stats['requests_this_hour']) >= key_config.get('max_requests_per_hour', 100) or
            len(stats['requests_this_day']) >= key_config.get('max_requests_per_day', 100)):
            return False
        
        return True
    
    def _disable_key_for_rate_limit(self, api_key: str, key_name: str):
        """Disable a key temporarily due to rate limiting"""
        disable_duration = min(300, 60 * (2 ** self.failure_counts.get(api_key, 0)))  # Exponential backoff
        self.disabled_keys[api_key] = time.time() + disable_duration
        logger.warning(f"Disabled API key {key_name} for {disable_duration} seconds due to rate limiting")
    
    def _calculate_key_score(self, key_config: dict) -> float:
        """Calculate a score for key selection (higher is better)"""
        api_key = key_config['api_key']
        current_time = time.time()
        
        # Base score from priority
        score = key_config.get('priority', 1) * 100
        
        # Penalize recently used keys
        if api_key in self.last_used:
            time_since_last_use = current_time - self.last_used[api_key]
            score += min(time_since_last_use / 60, 50)  # Bonus for keys not used recently
        
        # Penalize keys with high failure counts
        failure_count = self.failure_counts.get(api_key, 0)
        score -= failure_count * 10
        
        # Bonus for keys with low usage
        stats = self.usage_stats[api_key]
        usage_ratio = len(stats['requests_this_minute']) / key_config.get('max_requests_per_minute', 5)
        score += (1 - usage_ratio) * 20
        
        return score
    
    def get_available_api_key(self, use_random: bool = True) -> Optional[Tuple[str, dict]]:
        """Get an available API key with advanced selection logic"""
        available_keys = [
            key_config for key_config in self.api_keys 
            if self._is_key_available(key_config)
        ]
        
        if not available_keys:
            logger.error("No available API keys found")
            return None
        
        if use_random:
            # Weighted random selection based on scores
            scores = [self._calculate_key_score(key_config) for key_config in available_keys]
            total_score = sum(scores)
            if total_score > 0:
                weights = [score / total_score for score in scores]
                selected_key_config = random.choices(available_keys, weights=weights)[0]
            else:
                selected_key_config = random.choice(available_keys)
        else:
            # Select the key with the highest score
            selected_key_config = max(available_keys, key=lambda k: self._calculate_key_score(k))
        
        api_key = selected_key_config['api_key']
        self.last_used[api_key] = time.time()
        
        logger.info(f"Selected API key: {selected_key_config.get('name', 'Unknown')}")
        return api_key, selected_key_config
    
    def record_successful_request(self, api_key: str):
        """Record a successful API request"""
        if api_key in self.usage_stats:
            current_time = time.time()
            stats = self.usage_stats[api_key]
            stats['requests_this_minute'].append(current_time)
            stats['requests_this_hour'].append(current_time)
            stats['requests_this_day'].append(current_time)
            stats['total_requests'] += 1
            
            # Reset failure count on success
            if api_key in self.failure_counts:
                self.failure_counts[api_key] = 0
    
    def record_rate_limit_error(self, api_key: str, key_name: str):
        """Record a rate limit error"""
        self._disable_key_for_rate_limit(api_key, key_name)
        self.record_failure(api_key, key_name, "rate_limit")
    
    def record_failure(self, api_key: str, key_name: str, error_type: str = "unknown"):
        """Record an API failure"""
        if api_key in self.failure_counts:
            self.failure_counts[api_key] += 1
        
        logger.error(f"API key {key_name} failed with error type: {error_type}")
        
        # Disable key if too many consecutive failures
        if self.failure_counts.get(api_key, 0) >= 5:
            disable_duration = 3600  # 1 hour
            self.disabled_keys[api_key] = time.time() + disable_duration
            logger.warning(f"Disabled API key {key_name} for {disable_duration} seconds due to repeated failures")
    
    def get_keys_status(self) -> List[Dict]:
        """Get status of all API keys"""
        status_list = []
        current_time = time.time()
        
        for key_config in self.api_keys:
            api_key = key_config['api_key']
            stats = self.usage_stats.get(api_key, {})
            
            # Clean old data
            self._clean_old_usage_data(api_key)
            
            status = {
                'name': key_config.get('name', 'Unknown'),
                'enabled': key_config.get('enabled', True),
                'priority': key_config.get('priority', 1),
                'is_available': self._is_key_available(key_config),
                'is_disabled': api_key in self.disabled_keys,
                'disabled_until': self.disabled_keys.get(api_key),
                'failure_count': self.failure_counts.get(api_key, 0),
                'usage': {
                    'minute': len(stats.get('requests_this_minute', [])),
                    'hour': len(stats.get('requests_this_hour', [])),
                    'day': len(stats.get('requests_this_day', [])),
                    'total': stats.get('total_requests', 0)
                },
                'limits': {
                    'minute': key_config.get('max_requests_per_minute', 5),
                    'hour': key_config.get('max_requests_per_hour', 100),
                    'day': key_config.get('max_requests_per_day', 100)
                },
                'last_used': self.last_used.get(api_key)
            }
            status_list.append(status)
        
        return status_list

# Initialize API key manager
api_key_manager = AdvancedAPIKeyManager(config.get('api_keys', []))

# 初始化工作目录
# os.makedirs(app.config['WORKSPACE'], exist_ok=True) # This line is now handled by get_user_session
LOG_FILE = 'logs/root_stream.log'
FILE_CHECK_INTERVAL = 2  # 文件检查间隔（秒）
PROCESS_TIMEOUT = 6099999990    # 最长处理时间（秒）

def get_files_pathlib(root_dir):
    """Get all files in directory using pathlib"""
    files = []
    for path in Path(root_dir).rglob('*'):
        if path.is_file():
            files.append(str(path))
    return files

@app.route('/')
@require_user_session
def index():
    return render_template('index.html')

@app.route('/file/<filename>')
@require_user_session
def file(filename):
    user_session = get_user_session()
    file_path = os.path.join(user_session['uploads'], filename)
    
    if os.path.exists(file_path):
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        
        # For text files, return content
        if mime_type.startswith('text/') or mime_type in ['application/json', 'application/xml']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return jsonify({
                    'filename': filename,
                    'content': content,
                    'mime_type': mime_type,
                    'size': os.path.getsize(file_path)
                })
            except UnicodeDecodeError:
                return jsonify({'error': 'File contains binary data'}), 400
        
        # For binary files, return file info
        return jsonify({
            'filename': filename,
            'mime_type': mime_type,
            'size': os.path.getsize(file_path),
            'binary': True
        })
    
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/keys/status')
def api_keys_status():
    return jsonify(api_key_manager.get_keys_status())

# File upload utilities
def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'py', 'js', 'html', 'css', 'json', 'xml', 'csv', 'md', 'doc', 'docx', 'xls', 'xlsx', 'zip', 'rar', '7z'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_chat_history(chat_data):
    """Save chat history for current user"""
    user_session = get_user_session()
    user_session['chat_history'].append({
        'timestamp': datetime.now().isoformat(),
        'message': chat_data.get('message', ''),
        'response': chat_data.get('response', ''),
        'task_id': chat_data.get('task_id', ''),
        'files': chat_data.get('files', [])
    })
    
    # Keep only last 100 messages
    if len(user_session['chat_history']) > 100:
        user_session['chat_history'] = user_session['chat_history'][-100:]

def load_chat_history():
    """Load chat history for current user"""
    user_session = get_user_session()
    return user_session['chat_history']

@app.route('/api/upload', methods=['POST'])
@require_user_session
def upload_file():
    user_session = get_user_session()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}{ext}"
        
        file_path = os.path.join(user_session['uploads'], filename)
        file.save(file_path)
        
        # Get file info
        file_info = {
            'filename': filename,
            'size': os.path.getsize(file_path),
            'uploaded_at': datetime.now().isoformat(),
            'path': file_path
        }
        
        return jsonify({
            'message': 'File uploaded successfully',
            'file': file_info
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/github-clone', methods=['POST'])
@require_user_session
def github_clone():
    """Clone a GitHub repository"""
    user_session = get_user_session()
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'GitHub URL is required'}), 400
    
    github_url = data['url'].strip()
    
    # Validate GitHub URL
    if not (github_url.startswith('https://github.com/') or github_url.startswith('git@github.com:')):
        return jsonify({'error': 'Invalid GitHub URL'}), 400
    
    try:
        # Extract repo name from URL
        if github_url.startswith('https://github.com/'):
            repo_path = github_url.replace('https://github.com/', '').replace('.git', '')
        else:
            repo_path = github_url.replace('git@github.com:', '').replace('.git', '')
        
        repo_name = repo_path.replace('/', '_')
        clone_dir = os.path.join(user_session['github_clones'], repo_name)
        
        # Remove existing directory if it exists
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        
        # Clone the repository
        result = subprocess.run(
            ['git', 'clone', github_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            return jsonify({'error': f'Failed to clone repository: {result.stderr}'}), 400
        
        # Get repository info
        repo_info = {
            'name': repo_name,
            'path': clone_dir,
            'cloned_at': datetime.now().isoformat(),
            'files': get_files_pathlib(clone_dir)
        }
        
        return jsonify({
            'message': 'Repository cloned successfully',
            'repository': repo_info
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Clone operation timed out'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to clone repository: {str(e)}'}), 400

@app.route('/api/download-zip', methods=['POST'])
@require_user_session
def download_zip():
    """Create and download a ZIP file of selected files"""
    user_session = get_user_session()
    data = request.get_json()
    
    if not data or 'files' not in data:
        return jsonify({'error': 'Files list is required'}), 400
    
    files = data['files']
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Create temporary ZIP file
        zip_filename = f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(user_session['temp'], zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                if os.path.exists(file_path):
                    # Add file to ZIP with relative path
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
        
        return jsonify({
            'message': 'ZIP file created successfully',
            'zip_file': zip_filename,
            'download_url': f'/api/download/{zip_filename}'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to create ZIP file: {str(e)}'}), 400

@app.route('/api/download/<filename>')
@require_user_session
def download_file(filename):
    """Download a file from temp directory"""
    user_session = get_user_session()
    file_path = os.path.join(user_session['temp'], filename)
    
    if os.path.exists(file_path):
        return send_from_directory(user_session['temp'], filename, as_attachment=True)
    
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/files')
@require_user_session
def get_uploaded_files():
    user_session = get_user_session()
    files = []
    
    # Get uploaded files
    upload_dir = user_session['uploads']
    if os.path.exists(upload_dir):
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                files.append({
                    'name': filename,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'type': 'uploaded',
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
    
    # Get GitHub cloned repositories
    github_dir = user_session['github_clones']
    if os.path.exists(github_dir):
        for repo_name in os.listdir(github_dir):
            repo_path = os.path.join(github_dir, repo_name)
            if os.path.isdir(repo_path):
                repo_files = get_files_pathlib(repo_path)
                files.append({
                    'name': repo_name,
                    'path': repo_path,
                    'size': len(repo_files),
                    'type': 'github_repo',
                    'files': repo_files,
                    'modified': datetime.fromtimestamp(os.path.getmtime(repo_path)).isoformat()
                })
    
    return jsonify(files)

@app.route('/api/chat-history')
@require_user_session
def get_chat_history():
    return jsonify(load_chat_history())

@app.route('/api/stop-task', methods=['POST'])
@require_user_session
def stop_task():
    data = request.get_json()
    task_id = data.get('task_id')
    
    if task_id in running_tasks:
        # Signal the task to stop
        running_tasks[task_id]['stop_event'].set()
        del running_tasks[task_id]
        return jsonify({'message': 'Task stopped successfully'})
    
    return jsonify({'error': 'Task not found'}), 404

async def main(prompt, task_id=None):
    """Enhanced main function with advanced API key rotation and stop functionality"""
    max_retries = len(api_key_manager.api_keys)
    retry_count = 0
    
    while retry_count < max_retries:
        # Get available API key
        result = api_key_manager.get_available_api_key(use_random=True)
        if not result:
            logger.error("No API keys available for request")
            
            # Wait for next available key
            max_wait_time = 10  # 5 minutes
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
            
            # Create Manus agent with advanced API key manager
            agent = await Manus.create(
                api_key_manager=api_key_manager,
                api_key=api_key
            )
            
            # Execute the task
            await agent.run(prompt)
            
            # Record successful request
            api_key_manager.record_successful_request(api_key)
            logger.info(f"Task completed successfully with key: {key_name}")
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
                logger.error("All API keys exhausted, task failed")
                raise Exception(f"Task failed after trying all available API keys. Last error: {e}")
            
            logger.info(f"Retrying with different API key (attempt {retry_count + 1}/{max_retries})")
            
        finally:
            if 'agent' in locals():
                await agent.cleanup()

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

# WSGI entry point for deployment
application = app

if __name__ == '__main__':
    # Log initial API key status
    logger.info("=== Initial API Key Status ===")
    for status in api_key_manager.get_keys_status():
        logger.info(f"Key {status['name']}: Available={status['is_available']}, "
                    f"Usage={status['usage']['day']}/{status['limits']['day']} today")
    
    app.run(host='0.0.0.0', port=3000,  debug=False)
