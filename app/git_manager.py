"""
Git Manager for comprehensive Git operations
Handles cloning, branching, commits, and repository management
"""
import os
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from app.logger import logger

class GitManager:
    """Comprehensive Git operations manager"""
    
    def __init__(self, workspace_root: str = "workspace"):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(exist_ok=True)
        
    def clone_repository(self, repo_url: str, target_dir: Optional[str] = None, branch: Optional[str] = None) -> Dict:
        """Clone a Git repository"""
        try:
            if target_dir is None:
                repo_name = repo_url.split('/')[-1].replace('.git', '')
                target_dir = repo_name
            
            target_path = self.workspace_root / target_dir
            
            # Remove existing directory if it exists
            if target_path.exists():
                shutil.rmtree(target_path)
            
            # Build clone command
            cmd = ['git', 'clone']
            if branch:
                cmd.extend(['-b', branch])
            cmd.extend([repo_url, str(target_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned {repo_url} to {target_path}")
                
                # Get repository info
                repo_info = self.get_repository_info(str(target_path))
                
                return {
                    'success': True,
                    'message': f'Repository cloned successfully to {target_dir}',
                    'path': str(target_path),
                    'repo_info': repo_info
                }
            else:
                error_msg = result.stderr or result.stdout
                logger.error(f"Git clone failed: {error_msg}")
                return {
                    'success': False,
                    'error': f'Clone failed: {error_msg}'
                }
                
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Clone operation timed out'}
        except Exception as e:
            logger.error(f"Git clone error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_repository_info(self, repo_path: str) -> Dict:
        """Get comprehensive repository information"""
        try:
            repo_path = Path(repo_path)
            if not (repo_path / '.git').exists():
                return {'error': 'Not a Git repository'}
            
            info = {}
            
            # Get current branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'], 
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                info['current_branch'] = result.stdout.strip()
            
            # Get all branches
            result = subprocess.run(
                ['git', 'branch', '-a'], 
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                branches = [b.strip().replace('* ', '') for b in result.stdout.split('\n') if b.strip()]
                info['branches'] = branches
            
            # Get remote URL
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'], 
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                info['remote_url'] = result.stdout.strip()
            
            # Get last commit
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=format:%H|%an|%ae|%ad|%s'], 
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout:
                commit_parts = result.stdout.split('|')
                if len(commit_parts) == 5:
                    info['last_commit'] = {
                        'hash': commit_parts[0],
                        'author_name': commit_parts[1],
                        'author_email': commit_parts[2],
                        'date': commit_parts[3],
                        'message': commit_parts[4]
                    }
            
            # Get status
            result = subprocess.run(
                ['git', 'status', '--porcelain'], 
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                status_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                info['status'] = {
                    'clean': len(status_lines) == 0,
                    'modified_files': len([l for l in status_lines if l.startswith(' M')]),
                    'added_files': len([l for l in status_lines if l.startswith('A ')]),
                    'deleted_files': len([l for l in status_lines if l.startswith(' D')]),
                    'untracked_files': len([l for l in status_lines if l.startswith('??')])
                }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return {'error': str(e)}
    
    def create_branch(self, repo_path: str, branch_name: str, checkout: bool = True) -> Dict:
        """Create a new branch"""
        try:
            repo_path = Path(repo_path)
            if not (repo_path / '.git').exists():
                return {'success': False, 'error': 'Not a Git repository'}
            
            # Create branch
            cmd = ['git', 'branch', branch_name]
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr or result.stdout}
            
            # Checkout if requested
            if checkout:
                checkout_result = self.checkout_branch(str(repo_path), branch_name)
                if not checkout_result['success']:
                    return checkout_result
            
            return {
                'success': True,
                'message': f'Branch {branch_name} created successfully',
                'checked_out': checkout
            }
            
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return {'success': False, 'error': str(e)}
    
    def checkout_branch(self, repo_path: str, branch_name: str) -> Dict:
        """Checkout a branch"""
        try:
            repo_path = Path(repo_path)
            if not (repo_path / '.git').exists():
                return {'success': False, 'error': 'Not a Git repository'}
            
            result = subprocess.run(
                ['git', 'checkout', branch_name], 
                cwd=repo_path, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': f'Switched to branch {branch_name}'
                }
            else:
                return {'success': False, 'error': result.stderr or result.stdout}
                
        except Exception as e:
            logger.error(f"Error checking out branch: {e}")
            return {'success': False, 'error': str(e)}
    
    def commit_changes(self, repo_path: str, message: str, files: Optional[List[str]] = None) -> Dict:
        """Commit changes to the repository"""
        try:
            repo_path = Path(repo_path)
            if not (repo_path / '.git').exists():
                return {'success': False, 'error': 'Not a Git repository'}
            
            # Add files
            if files:
                for file in files:
                    result = subprocess.run(
                        ['git', 'add', file], 
                        cwd=repo_path, capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        return {'success': False, 'error': f'Failed to add {file}: {result.stderr}'}
            else:
                # Add all changes
                result = subprocess.run(
                    ['git', 'add', '.'], 
                    cwd=repo_path, capture_output=True, text=True
                )
                if result.returncode != 0:
                    return {'success': False, 'error': f'Failed to add files: {result.stderr}'}
            
            # Commit
            result = subprocess.run(
                ['git', 'commit', '-m', message], 
                cwd=repo_path, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': f'Changes committed successfully: {message}'
                }
            else:
                return {'success': False, 'error': result.stderr or result.stdout}
                
        except Exception as e:
            logger.error(f"Error committing changes: {e}")
            return {'success': False, 'error': str(e)}
    
    def push_changes(self, repo_path: str, remote: str = 'origin', branch: Optional[str] = None) -> Dict:
        """Push changes to remote repository"""
        try:
            repo_path = Path(repo_path)
            if not (repo_path / '.git').exists():
                return {'success': False, 'error': 'Not a Git repository'}
            
            cmd = ['git', 'push', remote]
            if branch:
                cmd.append(branch)
            
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': 'Changes pushed successfully'
                }
            else:
                return {'success': False, 'error': result.stderr or result.stdout}
                
        except Exception as e:
            logger.error(f"Error pushing changes: {e}")
            return {'success': False, 'error': str(e)}
    
    def pull_changes(self, repo_path: str, remote: str = 'origin', branch: Optional[str] = None) -> Dict:
        """Pull changes from remote repository"""
        try:
            repo_path = Path(repo_path)
            if not (repo_path / '.git').exists():
                return {'success': False, 'error': 'Not a Git repository'}
            
            cmd = ['git', 'pull', remote]
            if branch:
                cmd.append(branch)
            
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': 'Changes pulled successfully',
                    'output': result.stdout
                }
            else:
                return {'success': False, 'error': result.stderr or result.stdout}
                
        except Exception as e:
            logger.error(f"Error pulling changes: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_file_diff(self, repo_path: str, file_path: str, staged: bool = False) -> Dict:
        """Get diff for a specific file"""
        try:
            repo_path = Path(repo_path)
            if not (repo_path / '.git').exists():
                return {'success': False, 'error': 'Not a Git repository'}
            
            cmd = ['git', 'diff']
            if staged:
                cmd.append('--staged')
            cmd.append(file_path)
            
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
            
            return {
                'success': True,
                'diff': result.stdout,
                'has_changes': bool(result.stdout.strip())
            }
            
        except Exception as e:
            logger.error(f"Error getting file diff: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_commit_history(self, repo_path: str, limit: int = 10) -> Dict:
        """Get commit history"""
        try:
            repo_path = Path(repo_path)
            if not (repo_path / '.git').exists():
                return {'success': False, 'error': 'Not a Git repository'}
            
            result = subprocess.run(
                ['git', 'log', f'-{limit}', '--pretty=format:%H|%an|%ae|%ad|%s'], 
                cwd=repo_path, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                commits = []
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split('|')
                        if len(parts) == 5:
                            commits.append({
                                'hash': parts[0],
                                'author_name': parts[1],
                                'author_email': parts[2],
                                'date': parts[3],
                                'message': parts[4]
                            })
                
                return {'success': True, 'commits': commits}
            else:
                return {'success': False, 'error': result.stderr or result.stdout}
                
        except Exception as e:
            logger.error(f"Error getting commit history: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_repositories(self) -> List[Dict]:
        """List all Git repositories in workspace"""
        repositories = []
        
        try:
            for item in self.workspace_root.iterdir():
                if item.is_dir() and (item / '.git').exists():
                    repo_info = self.get_repository_info(str(item))
                    repo_info['name'] = item.name
                    repo_info['path'] = str(item)
                    repositories.append(repo_info)
            
            return repositories
            
        except Exception as e:
            logger.error(f"Error listing repositories: {e}")
            return []