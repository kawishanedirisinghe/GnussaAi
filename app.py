
# app.py
import os
import time
import json
import asyncio
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import mimetypes
from pathlib import Path

# Import the basic tool system
from app.tool.base import BaseTool
from app.logger import logger
from app.api_key_manager import APIKeyManager

app = Flask(__name__)
app.config['WORKSPACE'] = 'workspace'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize API key manager
api_key_manager = APIKeyManager()

# Initialize workspace directories
os.makedirs(app.config['WORKSPACE'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for task management
running_tasks = {}

def get_files_pathlib(root_dir):
    """Get files using pathlib recursively"""
    root = Path(root_dir)
    return [str(path) for path in root.glob('**/*') if path.is_file()]

@app.route('/')
def index():
    """Main page - simple file listing"""
    files = os.listdir(app.config['WORKSPACE'])
    return render_template('index.html', files=files)

@app.route('/file/<filename>')
def file(filename):
    """Serve individual files"""
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

# File upload utilities
def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'csv', 'xlsx', 'py', 'js', 'html', 'css', 'json', 'xml', 'md'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get file info
            file_size = os.path.getsize(filepath)
            file_info = {
                'filename': filename,
                'original_name': file.filename,
                'size': file_size,
                'upload_time': datetime.now().isoformat(),
                'path': filepath
            }
            
            return jsonify({
                'success': True,
                'file_info': file_info,
                'message': f'File {file.filename} uploaded successfully'
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/files')
def get_uploaded_files():
    """Get list of uploaded files"""
    try:
        files = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(filepath):
                    file_info = {
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'modified_time': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    }
                    files.append(file_info)
        
        return jsonify({'files': files})
        
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-task', methods=['POST'])
def stop_task():
    """Stop running AI task"""
    try:
        data = request.get_json()
        task_id = data.get('task_id', 'default')
        
        if task_id in running_tasks:
            # Signal the task to stop
            running_tasks[task_id]['stop_flag'] = True
            logger.info(f"Stop signal sent for task: {task_id}")
            return jsonify({'success': True, 'message': 'Stop signal sent'})
        else:
            return jsonify({'success': False, 'message': 'No running task found'})
            
    except Exception as e:
        logger.error(f"Error stopping task: {e}")
        return jsonify({'error': str(e)}), 500

def main(prompt, task_id=None):
    """Simple main function - basic functionality only"""
    try:
        logger.info(f"Processing simple query: {prompt}")
        
        # Simple response for now
        result = {
            'success': True,
            'message': f'Received your request: {prompt}',
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Thread wrapper for async tasks
def run_async_task(message, task_id=None):
    """Run task in thread"""
    try:
        result = main(message, task_id)
        if task_id and task_id in running_tasks:
            running_tasks[task_id]['result'] = result
            running_tasks[task_id]['status'] = 'completed'
    except Exception as e:
        logger.error(f"Error in async task: {e}")
        if task_id and task_id in running_tasks:
            running_tasks[task_id]['result'] = {'success': False, 'error': str(e)}
            running_tasks[task_id]['status'] = 'failed'

@app.route('/api/run-flow', methods=['POST'])
def run_flow_task():
    """Simple API endpoint for running tasks"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        task_id = data.get('task_id', f'task_{int(time.time())}')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Initialize task tracking
        running_tasks[task_id] = {
            'status': 'running',
            'start_time': time.time(),
            'stop_flag': False,
            'result': None
        }
        
        # Start task in separate thread
        thread = threading.Thread(target=run_async_task, args=(message, task_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Task started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting task: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/task-status/<task_id>')
def get_task_status(task_id):
    """Get status of a running task"""
    try:
        if task_id not in running_tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task_info = running_tasks[task_id]
        response = {
            'task_id': task_id,
            'status': task_info['status'],
            'start_time': task_info['start_time'],
            'elapsed_time': time.time() - task_info['start_time']
        }
        
        if task_info['result']:
            response['result'] = task_info['result']
        
        # Clean up completed tasks after returning result
        if task_info['status'] in ['completed', 'failed']:
            del running_tasks[task_id]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
