# OpenManus AI - Advanced IDE

A comprehensive VS Code-like development environment with AI assistance, featuring advanced coding capabilities, project templates, Git integration, and intelligent code modification.

## 🚀 Features

### 🎨 VS Code-like Interface
- **Activity Bar**: Quick access to Explorer, Search, Git, AI Assistant, and Templates
- **File Explorer**: Hierarchical file tree with context menus and file operations
- **Multi-tab Editor**: CodeMirror-based editor with syntax highlighting for multiple languages
- **Integrated Terminal**: Built-in terminal with command execution
- **Bottom Panel**: Terminal, AI Chat, and Output panels
- **Dark Theme**: Professional dark theme similar to VS Code

### 🤖 AI-Powered Features
- **Code Generation**: Generate code from natural language descriptions
- **Code Refactoring**: Intelligent code improvement and optimization
- **Code Explanation**: Detailed explanations of code functionality
- **Bug Fixing**: Automatic error detection and fixing
- **Language Conversion**: Convert code between different programming languages
- **Test Generation**: Automatic unit test creation
- **AI Chat Assistant**: Interactive coding help and guidance

### 📁 Advanced File Management
- **Directory Tree Navigation**: Expandable/collapsible folder structure
- **File Operations**: Create, delete, copy, move, rename files and folders
- **Multi-file Upload**: Drag and drop or browse multiple files
- **File Search**: Search by filename or content with regex support
- **Archive Support**: Extract and create ZIP/TAR archives
- **File Type Detection**: Automatic file type recognition and icons

### 🔧 Git Integration
- **Repository Cloning**: Clone repositories from any Git URL
- **Branch Management**: Create, switch, and manage branches
- **Commit Operations**: Stage and commit changes with messages
- **Push/Pull**: Sync with remote repositories
- **Repository Status**: View current branch, changes, and commit history
- **Multi-repository Support**: Manage multiple Git repositories

### 📋 Project Templates
- **Framework Templates**: Ready-to-use templates for popular frameworks
  - **Python**: Flask, FastAPI, Django
  - **JavaScript**: React, Express.js, Vue.js
  - **TypeScript**: React with TypeScript
  - **Java**: Spring Boot projects
- **Auto-scaffolding**: Complete project structure with dependencies
- **Customizable Variables**: Author info, project name, descriptions
- **Next Steps Guide**: Setup instructions for each template

### 🔍 Advanced Search
- **File Name Search**: Find files by name patterns
- **Content Search**: Search inside file contents
- **Case Sensitivity**: Toggle case-sensitive search
- **Regex Support**: Use regular expressions for complex searches
- **Search Results Navigation**: Click to open matching files

### 🛠️ Developer Tools
- **Syntax Highlighting**: Support for Python, JavaScript, TypeScript, HTML, CSS, JSON, Markdown
- **Auto-completion**: Intelligent code suggestions
- **Bracket Matching**: Automatic bracket and tag closing
- **Code Folding**: Collapse/expand code sections
- **Line Numbers**: Display line numbers in editor
- **Multiple Cursors**: Edit multiple locations simultaneously

### 🔐 Sandbox Environment
- **Secure Execution**: Isolated code execution environment
- **Resource Limits**: CPU, memory, and time constraints
- **Multiple Languages**: Support for various programming languages
- **Docker Integration**: Containerized execution for security

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git (for repository operations)
- Docker (optional, for enhanced sandbox features)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd openmanus-ai-ide
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   - Copy `config/config.toml.example` to `config/config.toml`
   - Add your LLM API keys (OpenAI, Anthropic, etc.)

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the IDE**:
   - Classic Interface: http://localhost:3000
   - Enhanced IDE: http://localhost:3000/ide

## 🎯 Usage Guide

### Getting Started
1. **Create a New Project**: Use the Templates panel to scaffold a new project
2. **Clone Repository**: Use Git integration to clone existing repositories  
3. **Upload Files**: Drag and drop files or use the upload button
4. **AI Generation**: Use AI Assistant to generate code from descriptions

### File Management
- **New File**: Click the + icon in Explorer or use Ctrl+N
- **New Folder**: Click the folder+ icon in Explorer
- **Context Menu**: Right-click files/folders for operations
- **Search**: Use the Search panel to find files and content

### Git Workflow
1. **Clone**: Enter repository URL in Git panel
2. **Branch**: Create and switch branches as needed
3. **Edit**: Make changes to files in the editor
4. **Commit**: Stage and commit changes with descriptive messages
5. **Push**: Sync changes to remote repository

### AI Assistance
- **Generate Code**: Describe what you want, select language, get code
- **Refactor**: Select code and choose refactoring options
- **Explain**: Get detailed explanations of complex code
- **Fix Bugs**: Paste error messages for automatic fixes
- **Chat**: Ask questions about coding concepts and best practices

### Project Templates
1. **Select Template**: Choose from available framework templates
2. **Configure**: Enter project name, author info, and options
3. **Generate**: Creates complete project structure with files
4. **Setup**: Follow the provided next steps for dependencies

## 🔧 Configuration

### API Keys
Configure your LLM provider API keys in `config/config.toml`:

```toml
[llm]
[[llm.api_keys]]
api_key = "your-openai-api-key"
name = "OpenAI Primary"
max_requests_per_minute = 60
max_requests_per_hour = 1000
max_requests_per_day = 10000
priority = 1
enabled = true

[[llm.api_keys]]
api_key = "your-anthropic-api-key"  
name = "Anthropic Claude"
max_requests_per_minute = 30
max_requests_per_hour = 500
max_requests_per_day = 5000
priority = 2
enabled = true
```

### Workspace Settings
- **WORKSPACE**: Directory for project files (default: `workspace/`)
- **UPLOAD_FOLDER**: Directory for uploaded files (default: `uploads/`)
- **MAX_CONTENT_LENGTH**: Maximum file upload size (default: 16MB)

## 🏗️ Architecture

### Backend Components
- **Flask Application**: Main web server and API endpoints
- **Git Manager**: Handles all Git operations and repository management
- **File Manager**: Advanced file system operations and management
- **AI Code Modifier**: Intelligent code analysis and modification
- **Project Templates**: Template system for project scaffolding
- **Sandbox Manager**: Secure code execution environment

### Frontend Components
- **Activity Bar**: Navigation and tool access
- **File Explorer**: Tree view with file operations
- **Code Editor**: CodeMirror-based editor with language support
- **Terminal**: Integrated command-line interface
- **AI Chat**: Interactive AI assistance panel
- **Modal System**: Dynamic dialogs for various operations

### API Endpoints

#### File Management
- `GET /api/files/tree` - Get directory structure
- `POST /api/files/create-directory` - Create new directory
- `DELETE /api/files/delete` - Delete files/directories
- `POST /api/files/copy` - Copy files/directories
- `POST /api/files/move` - Move/rename files/directories
- `GET /api/files/read` - Read file content
- `POST /api/files/write` - Write file content
- `GET /api/files/search` - Search files by name/content

#### Git Operations
- `GET /api/git/repositories` - List Git repositories
- `POST /api/git/clone` - Clone repository
- `GET /api/git/info/<path>` - Get repository information
- `POST /api/git/branch` - Create branch
- `POST /api/git/checkout` - Checkout branch
- `POST /api/git/commit` - Commit changes
- `POST /api/git/push` - Push to remote
- `POST /api/git/pull` - Pull from remote

#### AI Features
- `POST /api/ai/generate-code` - Generate code from prompt
- `POST /api/ai/refactor-code` - Refactor existing code
- `POST /api/ai/explain-code` - Explain code functionality
- `POST /api/ai/fix-code` - Fix code issues
- `POST /api/ai/convert-code` - Convert between languages
- `POST /api/ai/generate-tests` - Generate unit tests
- `POST /api/ai/modify-file` - AI-assisted file modification

#### Project Templates
- `GET /api/templates/list` - List available templates
- `POST /api/templates/create-project` - Create project from template

## 🎨 Customization

### Adding New Templates
1. Edit `app/project_templates.py`
2. Add template configuration to `self.templates`
3. Implement template methods following the existing pattern
4. Restart the application

### Adding Language Support
1. Add language mode to CodeMirror imports in `enhanced_ide.html`
2. Update language configurations in AI code modifier
3. Add file extension mappings for syntax highlighting
4. Update project templates if needed

### Theming
The interface uses CSS custom properties for easy theming. Modify the `:root` variables in `enhanced_ide.html` to customize colors:

```css
:root {
    --primary: #007acc;
    --dark: #1e1e1e;
    --text: #cccccc;
    /* ... other variables */
}
```

## 🧪 Testing

### Manual Testing
1. **File Operations**: Create, edit, delete files and folders
2. **Git Integration**: Clone repositories, make commits, push/pull
3. **AI Features**: Generate code, refactor, explain functionality
4. **Templates**: Create projects from different templates
5. **Search**: Test file and content search functionality

### API Testing
Use tools like Postman or curl to test API endpoints:

```bash
# Test file tree
curl http://localhost:3000/api/files/tree

# Test code generation
curl -X POST http://localhost:3000/api/ai/generate-code \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a hello world function", "language": "python"}'
```

## 🚀 Deployment

### Development
```bash
python app.py
```

### Production
1. **Use a WSGI server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:3000 app:application
   ```

2. **Set environment variables**:
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key
   ```

3. **Configure reverse proxy** (nginx/Apache) for static files and SSL

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 3000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:3000", "app:application"]
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes and test thoroughly**
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful commit messages
- Add documentation for new features
- Test all functionality before submitting
- Update README for significant changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CodeMirror** - Powerful code editor component
- **Font Awesome** - Beautiful icons
- **Flask** - Lightweight web framework
- **OpenAI/Anthropic** - AI language models for code assistance

## 🐛 Known Issues

- Terminal integration is simulated (not a real terminal)
- Some Git operations may require authentication setup
- Large file uploads may timeout on slower connections
- AI features require valid API keys and internet connection

## 🔮 Future Enhancements

- Real terminal integration with WebSocket
- Collaborative editing with multiple users
- Plugin system for custom extensions
- Integrated debugging capabilities
- Advanced Git visualization
- More project templates and languages
- Enhanced AI context awareness
- Performance optimizations for large projects

## 📞 Support

For support, feature requests, or bug reports:
1. **Check existing issues** on GitHub
2. **Create a new issue** with detailed description
3. **Include error messages** and steps to reproduce
4. **Specify your environment** (OS, Python version, etc.)

---

**OpenManus AI IDE** - Empowering developers with AI-assisted coding in a beautiful, VS Code-like environment. 🚀✨