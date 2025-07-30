
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
import threading
import toml
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

app = Flask(__name__)
app.config['WORKSPACE'] = 'workspace'

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

# 初始化工作目录
os.makedirs(app.config['WORKSPACE'], exist_ok=True)
LOG_FILE = 'logs/root_stream.log'
FILE_CHECK_INTERVAL = 2  # 文件检查间隔（秒）
PROCESS_TIMEOUT = 6099999990    # 最长处理时间（秒）

def get_files_pathlib(root_dir):
    """使用pathlib递归获取文件路径"""
    root = Path(root_dir)
    return [str(path) for path in root.glob('**/*') if path.is_file()]

@app.route('/')
def index():
    files = os.listdir(app.config['WORKSPACE'])
    return render_template('index.html', files=files)

@app.route('/file/<filename>')
def file(filename):
    file_path = os.path.join(app.config['WORKSPACE'], filename)
    if os.path.isfile(file_path):
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type and mime_type.startswith('text/'):
            if mime_type == 'text/html':
                return send_from_directory(app.config['WORKSPACE'], filename)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return render_template('code.html', filename=filename, content=content)
        elif mime_type == 'application/pdf':
            return send_from_directory(app.config['WORKSPACE'], filename)
        else:
            return send_from_directory(app.config['WORKSPACE'], filename)
    else:
        return "File not found", 404

@app.route('/api/keys/status')
def api_keys_status():
    """API endpoint to get status of all API keys"""
    return jsonify(api_key_manager.get_keys_status())

async def main(prompt):
    """Enhanced main function with advanced API key rotation"""
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

# 线程包装器
def run_async_task(message):
    """在新线程中运行异步任务"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(message))
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
    finally:
        loop.close()

@app.route('/api/chat-stream', methods=['POST'])
def chat_stream():
    """流式日志接口"""
    # 清空日志文件
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # 获取请求数据
    prompt = request.get_json()
    logger.info(f"收到请求: {prompt}")

    # 启动异步任务线程
    task_thread = threading.Thread(
        target=run_async_task,
        args=(prompt["message"],)
    )
    task_thread.start()

    # 流式生成器
    def generate():
        start_time = time.time()

        while task_thread.is_alive() or not log_queue.empty():
            # 超时检查
            if time.time() - start_time > PROCESS_TIMEOUT:
                yield """0303030"""
  
                break
            
            new_content = ""
            try:
                new_content = log_queue.get(timeout=0.1)
            except queue.Empty:
                pass

            if new_content:
                yield new_content

            # 无新内容时暂停
            if not new_content:
                time.sleep(FILE_CHECK_INTERVAL)

        # 最终确认
        yield """0303030"""

    return Response(generate(), mimetype="text/plain")

# Run flow async task
async def run_flow_task(prompt):
    """Enhanced run_flow function with advanced API key rotation"""
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

def run_flow_async_task(message):
    """在新线程中运行flow异步任务"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_flow_task(message))
    except Exception as e:
        logger.error(f"Flow task execution failed: {e}")
    finally:
        loop.close()

@app.route('/api/flow-stream', methods=['POST'])
def flow_stream():
    """Flow流式日志接口"""
    # 清空日志文件
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # 获取请求数据
    prompt = request.get_json()
    logger.info(f"收到Flow请求: {prompt}")

    # 启动异步任务线程
    task_thread = threading.Thread(
        target=run_flow_async_task,
        args=(prompt["message"],)
    )
    task_thread.start()

    # 流式生成器
    def generate():
        start_time = time.time()

        while task_thread.is_alive() or not log_queue.empty():
            # 超时检查
            if time.time() - start_time > PROCESS_TIMEOUT:
                yield """0303030"""
                break
            
            new_content = ""
            try:
                new_content = log_queue.get(timeout=0.1)
            except queue.Empty:
                pass

            if new_content:
                yield new_content

            # 无新内容时暂停
            if not new_content:
                time.sleep(FILE_CHECK_INTERVAL)

        # 最终确认
        yield """0303030"""

    return Response(generate(), mimetype="text/plain")

# WSGI entry point for deployment
application = app

if __name__ == '__main__':
    # Log initial API key status
    logger.info("=== Initial API Key Status ===")
    for status in api_key_manager.get_keys_status():
        logger.info(f"Key {status['name']}: Available={status['available']}, "
                    f"Usage={status['usage']['requests_this_day']}/{status['limits']['max_per_day']} today")
    
    app.run(host='0.0.0.0', port=3000,  debug=False)
