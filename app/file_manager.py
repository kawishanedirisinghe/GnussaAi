"""
Advanced File Manager for comprehensive file operations
Handles file/directory operations, navigation, and multi-file management
"""
import os
import shutil
import mimetypes
import zipfile
import tarfile
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
import hashlib
# import magic  # Optional dependency for better file type detection
from app.logger import logger

class AdvancedFileManager:
    """Comprehensive file and directory management system"""
    
    def __init__(self, workspace_root: str = "workspace", upload_folder: str = "uploads"):
        self.workspace_root = Path(workspace_root)
        self.upload_folder = Path(upload_folder)
        self.workspace_root.mkdir(exist_ok=True)
        self.upload_folder.mkdir(exist_ok=True)
        
        # Supported file types for various operations
        self.text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico'}
        self.archive_extensions = {'.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.rar', '.7z'}
        self.code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.kt', '.swift'}
    
    def get_directory_tree(self, path: str = None, max_depth: int = 3, current_depth: int = 0) -> Dict:
        """Get comprehensive directory tree structure"""
        if path is None:
            path = str(self.workspace_root)
        
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return {'error': 'Path does not exist'}
            
            if not path_obj.is_dir():
                return {'error': 'Path is not a directory'}
            
            tree = {
                'name': path_obj.name or str(path_obj),
                'path': str(path_obj),
                'type': 'directory',
                'size': 0,
                'modified': datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat(),
                'children': []
            }
            
            if current_depth >= max_depth:
                tree['truncated'] = True
                return tree
            
            total_size = 0
            for item in sorted(path_obj.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                try:
                    item_stat = item.stat()
                    item_info = {
                        'name': item.name,
                        'path': str(item),
                        'type': 'directory' if item.is_dir() else 'file',
                        'size': item_stat.st_size,
                        'modified': datetime.fromtimestamp(item_stat.st_mtime).isoformat(),
                        'permissions': oct(item_stat.st_mode)[-3:]
                    }
                    
                    if item.is_file():
                        # Add file-specific information
                        item_info['extension'] = item.suffix.lower()
                        item_info['mime_type'] = mimetypes.guess_type(str(item))[0]
                        item_info['is_text'] = item.suffix.lower() in self.text_extensions
                        item_info['is_image'] = item.suffix.lower() in self.image_extensions
                        item_info['is_archive'] = item.suffix.lower() in self.archive_extensions
                        item_info['is_code'] = item.suffix.lower() in self.code_extensions
                        total_size += item_stat.st_size
                    elif item.is_dir() and current_depth < max_depth:
                        # Recursively get subdirectory info
                        subtree = self.get_directory_tree(str(item), max_depth, current_depth + 1)
                        if 'children' in subtree:
                            item_info['children'] = subtree['children']
                            item_info['size'] = subtree['size']
                            total_size += subtree['size']
                    
                    tree['children'].append(item_info)
                    
                except (PermissionError, OSError) as e:
                    logger.warning(f"Cannot access {item}: {e}")
                    continue
            
            tree['size'] = total_size
            return tree
            
        except Exception as e:
            logger.error(f"Error getting directory tree: {e}")
            return {'error': str(e)}
    
    def create_directory(self, path: str, parents: bool = True) -> Dict:
        """Create a new directory"""
        try:
            dir_path = Path(path)
            if dir_path.is_absolute():
                # Ensure it's within workspace for security
                if not str(dir_path).startswith(str(self.workspace_root)):
                    return {'success': False, 'error': 'Path outside workspace not allowed'}
            else:
                dir_path = self.workspace_root / path
            
            dir_path.mkdir(parents=parents, exist_ok=False)
            
            return {
                'success': True,
                'message': f'Directory created: {dir_path.name}',
                'path': str(dir_path)
            }
            
        except FileExistsError:
            return {'success': False, 'error': 'Directory already exists'}
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            return {'success': False, 'error': str(e)}
    
    def delete_item(self, path: str, force: bool = False) -> Dict:
        """Delete a file or directory"""
        try:
            item_path = Path(path)
            if item_path.is_absolute():
                # Ensure it's within workspace for security
                if not str(item_path).startswith(str(self.workspace_root)):
                    return {'success': False, 'error': 'Path outside workspace not allowed'}
            else:
                item_path = self.workspace_root / path
            
            if not item_path.exists():
                return {'success': False, 'error': 'Item does not exist'}
            
            if item_path.is_file():
                item_path.unlink()
                return {
                    'success': True,
                    'message': f'File deleted: {item_path.name}'
                }
            elif item_path.is_dir():
                if force or not any(item_path.iterdir()):
                    shutil.rmtree(item_path)
                    return {
                        'success': True,
                        'message': f'Directory deleted: {item_path.name}'
                    }
                else:
                    return {'success': False, 'error': 'Directory not empty. Use force=True to delete.'}
            
        except Exception as e:
            logger.error(f"Error deleting item: {e}")
            return {'success': False, 'error': str(e)}
    
    def copy_item(self, source: str, destination: str, overwrite: bool = False) -> Dict:
        """Copy a file or directory"""
        try:
            src_path = Path(source)
            dst_path = Path(destination)
            
            # Handle relative paths
            if not src_path.is_absolute():
                src_path = self.workspace_root / source
            if not dst_path.is_absolute():
                dst_path = self.workspace_root / destination
            
            # Security check
            for path in [src_path, dst_path]:
                if not str(path).startswith(str(self.workspace_root)):
                    return {'success': False, 'error': 'Path outside workspace not allowed'}
            
            if not src_path.exists():
                return {'success': False, 'error': 'Source does not exist'}
            
            if dst_path.exists() and not overwrite:
                return {'success': False, 'error': 'Destination exists. Use overwrite=True to replace.'}
            
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
                return {
                    'success': True,
                    'message': f'File copied: {src_path.name} -> {dst_path.name}'
                }
            elif src_path.is_dir():
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                return {
                    'success': True,
                    'message': f'Directory copied: {src_path.name} -> {dst_path.name}'
                }
            
        except Exception as e:
            logger.error(f"Error copying item: {e}")
            return {'success': False, 'error': str(e)}
    
    def move_item(self, source: str, destination: str, overwrite: bool = False) -> Dict:
        """Move/rename a file or directory"""
        try:
            src_path = Path(source)
            dst_path = Path(destination)
            
            # Handle relative paths
            if not src_path.is_absolute():
                src_path = self.workspace_root / source
            if not dst_path.is_absolute():
                dst_path = self.workspace_root / destination
            
            # Security check
            for path in [src_path, dst_path]:
                if not str(path).startswith(str(self.workspace_root)):
                    return {'success': False, 'error': 'Path outside workspace not allowed'}
            
            if not src_path.exists():
                return {'success': False, 'error': 'Source does not exist'}
            
            if dst_path.exists() and not overwrite:
                return {'success': False, 'error': 'Destination exists. Use overwrite=True to replace.'}
            
            if dst_path.exists() and overwrite:
                if dst_path.is_file():
                    dst_path.unlink()
                else:
                    shutil.rmtree(dst_path)
            
            shutil.move(str(src_path), str(dst_path))
            
            return {
                'success': True,
                'message': f'Item moved: {src_path.name} -> {dst_path.name}'
            }
            
        except Exception as e:
            logger.error(f"Error moving item: {e}")
            return {'success': False, 'error': str(e)}
    
    def read_file(self, file_path: str, encoding: str = 'utf-8', max_size: int = 10*1024*1024) -> Dict:
        """Read file content with size and encoding handling"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.workspace_root / file_path
            
            # Security check
            if not str(path).startswith(str(self.workspace_root)):
                return {'success': False, 'error': 'Path outside workspace not allowed'}
            
            if not path.exists():
                return {'success': False, 'error': 'File does not exist'}
            
            if not path.is_file():
                return {'success': False, 'error': 'Path is not a file'}
            
            file_size = path.stat().st_size
            if file_size > max_size:
                return {
                    'success': False, 
                    'error': f'File too large ({file_size} bytes). Maximum size: {max_size} bytes'
                }
            
            # Try to determine file type
            mime_type = mimetypes.guess_type(str(path))[0]
            is_binary = False
            
            try:
                # Try reading as text first
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # If text reading fails, it's likely binary
                is_binary = True
                with open(path, 'rb') as f:
                    binary_content = f.read()
                    # For binary files, return base64 encoded content
                    import base64
                    content = base64.b64encode(binary_content).decode('ascii')
            
            return {
                'success': True,
                'content': content,
                'size': file_size,
                'mime_type': mime_type,
                'is_binary': is_binary,
                'encoding': encoding if not is_binary else 'base64',
                'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {'success': False, 'error': str(e)}
    
    def write_file(self, file_path: str, content: str, encoding: str = 'utf-8', create_dirs: bool = True) -> Dict:
        """Write content to a file"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.workspace_root / file_path
            
            # Security check
            if not str(path).startswith(str(self.workspace_root)):
                return {'success': False, 'error': 'Path outside workspace not allowed'}
            
            # Create parent directories if needed
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            file_size = path.stat().st_size
            
            return {
                'success': True,
                'message': f'File written: {path.name}',
                'path': str(path),
                'size': file_size
            }
            
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            return {'success': False, 'error': str(e)}
    
    def search_files(self, pattern: str, search_path: str = None, case_sensitive: bool = False, 
                    file_content: bool = False, max_results: int = 100) -> Dict:
        """Search for files by name or content"""
        try:
            if search_path is None:
                search_path = str(self.workspace_root)
            
            search_root = Path(search_path)
            if not search_root.is_absolute():
                search_root = self.workspace_root / search_path
            
            # Security check
            if not str(search_root).startswith(str(self.workspace_root)):
                return {'success': False, 'error': 'Search path outside workspace not allowed'}
            
            results = []
            
            if not case_sensitive:
                pattern = pattern.lower()
            
            for file_path in search_root.rglob('*'):
                if len(results) >= max_results:
                    break
                
                try:
                    if file_path.is_file():
                        file_name = file_path.name if case_sensitive else file_path.name.lower()
                        
                        # Search by filename
                        if pattern in file_name:
                            results.append({
                                'path': str(file_path),
                                'name': file_path.name,
                                'size': file_path.stat().st_size,
                                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                                'match_type': 'filename'
                            })
                        
                        # Search by content if requested and file is text
                        elif file_content and file_path.suffix.lower() in self.text_extensions:
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    search_content = content if case_sensitive else content.lower()
                                    if pattern in search_content:
                                        # Find line numbers where pattern occurs
                                        lines = content.split('\n')
                                        matching_lines = []
                                        for i, line in enumerate(lines, 1):
                                            search_line = line if case_sensitive else line.lower()
                                            if pattern in search_line:
                                                matching_lines.append({'line_number': i, 'content': line.strip()})
                                        
                                        results.append({
                                            'path': str(file_path),
                                            'name': file_path.name,
                                            'size': file_path.stat().st_size,
                                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                                            'match_type': 'content',
                                            'matching_lines': matching_lines[:10]  # Limit to first 10 matches
                                        })
                            except (UnicodeDecodeError, PermissionError):
                                continue
                
                except (PermissionError, OSError):
                    continue
            
            return {
                'success': True,
                'results': results,
                'total_found': len(results),
                'truncated': len(results) >= max_results
            }
            
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_file_info(self, file_path: str) -> Dict:
        """Get comprehensive file information"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.workspace_root / file_path
            
            # Security check
            if not str(path).startswith(str(self.workspace_root)):
                return {'success': False, 'error': 'Path outside workspace not allowed'}
            
            if not path.exists():
                return {'success': False, 'error': 'File does not exist'}
            
            stat_info = path.stat()
            
            info = {
                'name': path.name,
                'path': str(path),
                'size': stat_info.st_size,
                'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                'permissions': oct(stat_info.st_mode)[-3:],
                'is_file': path.is_file(),
                'is_directory': path.is_dir(),
                'extension': path.suffix.lower() if path.is_file() else None,
                'mime_type': mimetypes.guess_type(str(path))[0] if path.is_file() else None
            }
            
            if path.is_file():
                info['is_text'] = path.suffix.lower() in self.text_extensions
                info['is_image'] = path.suffix.lower() in self.image_extensions
                info['is_archive'] = path.suffix.lower() in self.archive_extensions
                info['is_code'] = path.suffix.lower() in self.code_extensions
                
                # Calculate file hash for integrity checking
                if stat_info.st_size < 50 * 1024 * 1024:  # Only for files < 50MB
                    try:
                        with open(path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            info['md5_hash'] = file_hash
                    except Exception:
                        pass
            
            return {'success': True, 'info': info}
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {'success': False, 'error': str(e)}
    
    def extract_archive(self, archive_path: str, destination: str = None) -> Dict:
        """Extract archive files (zip, tar, etc.)"""
        try:
            archive = Path(archive_path)
            if not archive.is_absolute():
                archive = self.workspace_root / archive_path
            
            if destination is None:
                destination = archive.parent / archive.stem
            else:
                destination = Path(destination)
                if not destination.is_absolute():
                    destination = self.workspace_root / destination
            
            # Security checks
            for path in [archive, destination]:
                if not str(path).startswith(str(self.workspace_root)):
                    return {'success': False, 'error': 'Path outside workspace not allowed'}
            
            if not archive.exists():
                return {'success': False, 'error': 'Archive file does not exist'}
            
            # Create destination directory
            destination.mkdir(parents=True, exist_ok=True)
            
            extracted_files = []
            
            if archive.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive, 'r') as zip_ref:
                    zip_ref.extractall(destination)
                    extracted_files = zip_ref.namelist()
            
            elif archive.suffix.lower() in ['.tar', '.tar.gz', '.tgz', '.tar.bz2']:
                with tarfile.open(archive, 'r:*') as tar_ref:
                    tar_ref.extractall(destination)
                    extracted_files = tar_ref.getnames()
            
            else:
                return {'success': False, 'error': 'Unsupported archive format'}
            
            return {
                'success': True,
                'message': f'Archive extracted to {destination.name}',
                'destination': str(destination),
                'extracted_files': extracted_files[:50]  # Limit output
            }
            
        except Exception as e:
            logger.error(f"Error extracting archive: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_archive(self, source_path: str, archive_path: str, archive_type: str = 'zip') -> Dict:
        """Create archive from files/directories"""
        try:
            source = Path(source_path)
            archive = Path(archive_path)
            
            if not source.is_absolute():
                source = self.workspace_root / source_path
            if not archive.is_absolute():
                archive = self.workspace_root / archive_path
            
            # Security checks
            for path in [source, archive]:
                if not str(path).startswith(str(self.workspace_root)):
                    return {'success': False, 'error': 'Path outside workspace not allowed'}
            
            if not source.exists():
                return {'success': False, 'error': 'Source does not exist'}
            
            # Create parent directory for archive
            archive.parent.mkdir(parents=True, exist_ok=True)
            
            if archive_type.lower() == 'zip':
                with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    if source.is_file():
                        zip_ref.write(source, source.name)
                    else:
                        for file_path in source.rglob('*'):
                            if file_path.is_file():
                                arc_name = file_path.relative_to(source)
                                zip_ref.write(file_path, arc_name)
            
            elif archive_type.lower() in ['tar', 'tar.gz', 'tgz']:
                mode = 'w:gz' if archive_type.lower() in ['tar.gz', 'tgz'] else 'w'
                with tarfile.open(archive, mode) as tar_ref:
                    tar_ref.add(source, arcname=source.name)
            
            else:
                return {'success': False, 'error': 'Unsupported archive type'}
            
            archive_size = archive.stat().st_size
            
            return {
                'success': True,
                'message': f'Archive created: {archive.name}',
                'archive_path': str(archive),
                'size': archive_size
            }
            
        except Exception as e:
            logger.error(f"Error creating archive: {e}")
            return {'success': False, 'error': str(e)}