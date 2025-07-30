"""
Project Template System
Provides scaffolding for common frameworks and project structures
"""
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import shutil
from app.logger import logger

class ProjectTemplateManager:
    """Manages project templates and scaffolding"""
    
    def __init__(self, workspace_root: str = "workspace"):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(exist_ok=True)
        
        # Define built-in templates
        self.templates = {
            'python_flask': {
                'name': 'Flask Web Application',
                'description': 'A complete Flask web application with modern structure',
                'language': 'python',
                'framework': 'flask',
                'files': {
                    'app.py': self._get_flask_app_template(),
                    'requirements.txt': self._get_flask_requirements(),
                    'config.py': self._get_flask_config_template(),
                    'models.py': self._get_flask_models_template(),
                    'routes.py': self._get_flask_routes_template(),
                    'templates/base.html': self._get_flask_base_template(),
                    'templates/index.html': self._get_flask_index_template(),
                    'static/css/style.css': self._get_flask_css_template(),
                    'static/js/app.js': self._get_flask_js_template(),
                    'tests/test_app.py': self._get_flask_test_template(),
                    'README.md': self._get_flask_readme_template(),
                    '.env.example': self._get_flask_env_template(),
                    '.gitignore': self._get_python_gitignore()
                }
            },
            'python_fastapi': {
                'name': 'FastAPI Application',
                'description': 'Modern FastAPI application with async support',
                'language': 'python',
                'framework': 'fastapi',
                'files': {
                    'main.py': self._get_fastapi_main_template(),
                    'requirements.txt': self._get_fastapi_requirements(),
                    'models.py': self._get_fastapi_models_template(),
                    'database.py': self._get_fastapi_database_template(),
                    'routers/users.py': self._get_fastapi_users_router(),
                    'tests/test_main.py': self._get_fastapi_test_template(),
                    'README.md': self._get_fastapi_readme_template(),
                    '.env.example': self._get_fastapi_env_template(),
                    '.gitignore': self._get_python_gitignore()
                }
            },
            'react_app': {
                'name': 'React Application',
                'description': 'Modern React application with hooks and routing',
                'language': 'javascript',
                'framework': 'react',
                'files': {
                    'package.json': self._get_react_package_json(),
                    'src/App.js': self._get_react_app_template(),
                    'src/index.js': self._get_react_index_template(),
                    'src/components/Header.js': self._get_react_header_component(),
                    'src/components/Footer.js': self._get_react_footer_component(),
                    'src/pages/Home.js': self._get_react_home_page(),
                    'src/pages/About.js': self._get_react_about_page(),
                    'src/styles/App.css': self._get_react_css_template(),
                    'public/index.html': self._get_react_html_template(),
                    'README.md': self._get_react_readme_template(),
                    '.gitignore': self._get_node_gitignore()
                }
            },
            'node_express': {
                'name': 'Express.js API',
                'description': 'RESTful API with Express.js and middleware',
                'language': 'javascript',
                'framework': 'express',
                'files': {
                    'package.json': self._get_express_package_json(),
                    'server.js': self._get_express_server_template(),
                    'routes/api.js': self._get_express_routes_template(),
                    'middleware/auth.js': self._get_express_auth_middleware(),
                    'models/User.js': self._get_express_user_model(),
                    'config/database.js': self._get_express_db_config(),
                    'tests/api.test.js': self._get_express_test_template(),
                    'README.md': self._get_express_readme_template(),
                    '.env.example': self._get_express_env_template(),
                    '.gitignore': self._get_node_gitignore()
                }
            },
            'vue_app': {
                'name': 'Vue.js Application',
                'description': 'Vue.js application with Vuex and Vue Router',
                'language': 'javascript',
                'framework': 'vue',
                'files': {
                    'package.json': self._get_vue_package_json(),
                    'src/main.js': self._get_vue_main_template(),
                    'src/App.vue': self._get_vue_app_template(),
                    'src/components/HelloWorld.vue': self._get_vue_component_template(),
                    'src/views/Home.vue': self._get_vue_home_view(),
                    'src/views/About.vue': self._get_vue_about_view(),
                    'src/router/index.js': self._get_vue_router_template(),
                    'src/store/index.js': self._get_vue_store_template(),
                    'public/index.html': self._get_vue_html_template(),
                    'README.md': self._get_vue_readme_template(),
                    '.gitignore': self._get_node_gitignore()
                }
            },
            'django_app': {
                'name': 'Django Application',
                'description': 'Django web application with models and views',
                'language': 'python',
                'framework': 'django',
                'files': {
                    'manage.py': self._get_django_manage_template(),
                    'requirements.txt': self._get_django_requirements(),
                    'myproject/settings.py': self._get_django_settings_template(),
                    'myproject/urls.py': self._get_django_urls_template(),
                    'myproject/wsgi.py': self._get_django_wsgi_template(),
                    'myapp/models.py': self._get_django_models_template(),
                    'myapp/views.py': self._get_django_views_template(),
                    'myapp/urls.py': self._get_django_app_urls_template(),
                    'templates/base.html': self._get_django_base_template(),
                    'templates/index.html': self._get_django_index_template(),
                    'static/css/style.css': self._get_django_css_template(),
                    'README.md': self._get_django_readme_template(),
                    '.gitignore': self._get_python_gitignore()
                }
            }
        }
    
    def list_templates(self) -> List[Dict]:
        """List all available project templates"""
        return [
            {
                'id': template_id,
                'name': template['name'],
                'description': template['description'],
                'language': template['language'],
                'framework': template['framework']
            }
            for template_id, template in self.templates.items()
        ]
    
    def create_project(self, template_id: str, project_name: str, 
                      custom_vars: Optional[Dict] = None) -> Dict:
        """Create a new project from template"""
        try:
            if template_id not in self.templates:
                return {'success': False, 'error': f'Template {template_id} not found'}
            
            template = self.templates[template_id]
            project_path = self.workspace_root / project_name
            
            if project_path.exists():
                return {'success': False, 'error': f'Project {project_name} already exists'}
            
            # Create project directory
            project_path.mkdir(parents=True)
            
            # Template variables
            template_vars = {
                'project_name': project_name,
                'author': 'Developer',
                'email': 'developer@example.com',
                'description': f'A {template["name"]} project',
                'version': '1.0.0',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'year': datetime.now().year
            }
            
            if custom_vars:
                template_vars.update(custom_vars)
            
            created_files = []
            
            # Create files from template
            for file_path, content in template['files'].items():
                full_path = project_path / file_path
                
                # Create parent directories
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Replace template variables
                processed_content = self._process_template_vars(content, template_vars)
                
                # Write file
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)
                
                created_files.append(str(full_path.relative_to(project_path)))
            
            return {
                'success': True,
                'message': f'Project {project_name} created successfully',
                'project_path': str(project_path),
                'template_used': template_id,
                'files_created': created_files,
                'next_steps': self._get_next_steps(template_id, project_name)
            }
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_template_vars(self, content: str, variables: Dict) -> str:
        """Replace template variables in content"""
        for var, value in variables.items():
            content = content.replace(f'{{{{{var}}}}}', str(value))
        return content
    
    def _get_next_steps(self, template_id: str, project_name: str) -> List[str]:
        """Get next steps for the created project"""
        steps = [f"cd {project_name}"]
        
        if template_id.startswith('python_'):
            steps.extend([
                "python -m venv venv",
                "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
                "pip install -r requirements.txt"
            ])
            
            if template_id == 'python_flask':
                steps.append("python app.py")
            elif template_id == 'python_fastapi':
                steps.append("uvicorn main:app --reload")
            elif template_id == 'django_app':
                steps.extend([
                    "python manage.py migrate",
                    "python manage.py runserver"
                ])
        
        elif template_id.startswith('react_') or template_id.startswith('node_') or template_id.startswith('vue_'):
            steps.extend([
                "npm install",
                "npm start"
            ])
        
        return steps
    
    # Flask Templates
    def _get_flask_app_template(self) -> str:
        return '''from flask import Flask, render_template, request, jsonify
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
def index():
    return render_template('index.html', title='{{project_name}}')

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

    def _get_flask_requirements(self) -> str:
        return '''Flask==2.3.3
python-dotenv==1.0.0
Flask-SQLAlchemy==3.0.5
Flask-Migrate==4.0.5
Flask-CORS==4.0.0
'''

    def _get_flask_config_template(self) -> str:
        return '''import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
'''

    def _get_flask_models_template(self) -> str:
        return '''from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }
'''

    def _get_flask_routes_template(self) -> str:
        return '''from flask import Blueprint, request, jsonify
from models import User, db

api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@api.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    
    user = User(
        username=data.get('username'),
        email=data.get('email')
    )
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify(user.to_dict()), 201
'''

    def _get_flask_base_template(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{{project_name}}{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">{{project_name}}</a>
        </div>
    </nav>
    
    <main class="container mt-4">
        {% block content %}{% endblock %}
    </main>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>
'''

    def _get_flask_index_template(self) -> str:
        return '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-8 mx-auto">
        <div class="jumbotron bg-light p-5 rounded">
            <h1 class="display-4">Welcome to {{project_name}}!</h1>
            <p class="lead">This is a Flask application created from template.</p>
            <hr class="my-4">
            <p>Get started by editing the templates and routes.</p>
            <a class="btn btn-primary btn-lg" href="/api/health" role="button">Test API</a>
        </div>
    </div>
</div>
{% endblock %}
'''

    def _get_flask_css_template(self) -> str:
        return '''/* Custom styles for {{project_name}} */

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}

.jumbotron {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.navbar-brand {
    font-weight: bold;
}

.container {
    max-width: 1200px;
}
'''

    def _get_flask_js_template(self) -> str:
        return '''// JavaScript for {{project_name}}

document.addEventListener('DOMContentLoaded', function() {
    console.log('{{project_name}} loaded successfully!');
    
    // Add your JavaScript code here
});

// Example API call function
async function testAPI() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('API Response:', data);
        return data;
    } catch (error) {
        console.error('API Error:', error);
    }
}
'''

    def _get_flask_test_template(self) -> str:
        return '''import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test the index route"""
    response = client.get('/')
    assert response.status_code == 200

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
'''

    def _get_flask_readme_template(self) -> str:
        return '''# {{project_name}}

A Flask web application created on {{date}}.

## Description

{{description}}

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

Run the development server:
```bash
python app.py
```

The application will be available at http://localhost:5000

## API Endpoints

- `GET /` - Home page
- `GET /api/health` - Health check endpoint

## Testing

Run tests with:
```bash
pytest
```

## Author

{{author}} ({{email}})

## License

MIT License
'''

    def _get_flask_env_template(self) -> str:
        return '''# Flask Configuration
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///app.db
FLASK_ENV=development
'''

    # FastAPI Templates
    def _get_fastapi_main_template(self) -> str:
        return '''from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(
    title="{{project_name}}",
    description="{{description}}",
    version="{{version}}"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=dict)
async def root():
    return {"message": "Welcome to {{project_name}}!"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="API is running")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    def _get_fastapi_requirements(self) -> str:
        return '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
python-dotenv==1.0.0
sqlalchemy==2.0.23
'''

    def _get_fastapi_models_template(self) -> str:
        return '''from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
'''

    def _get_fastapi_database_template(self) -> str:
        return '''from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
'''

    def _get_fastapi_users_router(self) -> str:
        return '''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database import get_db, User as DBUser
from models import User, UserCreate, UserResponse

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/", response_model=List[UserResponse])
async def get_users(db: Session = Depends(get_db)):
    users = db.query(DBUser).all()
    return users

@router.post("/", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = DBUser(username=user.username, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
'''

    def _get_fastapi_test_template(self) -> str:
        return '''from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
'''

    def _get_fastapi_readme_template(self) -> str:
        return '''# {{project_name}}

A FastAPI application created on {{date}}.

## Description

{{description}}

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the development server:
```bash
uvicorn main:app --reload
```

The application will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint
- `GET /users` - Get all users
- `POST /users` - Create a new user
- `GET /users/{user_id}` - Get user by ID

## Testing

Run tests with:
```bash
pytest
```

## Author

{{author}} ({{email}})

## License

MIT License
'''

    def _get_fastapi_env_template(self) -> str:
        return '''# FastAPI Configuration
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=your-secret-key-here
'''

    # Common templates
    def _get_python_gitignore(self) -> str:
        return '''# Python
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
'''

    def _get_node_gitignore(self) -> str:
        return '''# Dependencies
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
'''

    # React Templates
    def _get_react_package_json(self) -> str:
        return '''{
  "name": "{{project_name}}",
  "version": "{{version}}",
  "description": "{{description}}",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "author": "{{author}}",
  "license": "MIT"
}
'''

    def _get_react_app_template(self) -> str:
        return '''import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import About from './pages/About';
import './styles/App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
'''

    def _get_react_index_template(self) -> str:
        return '''import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
'''

    def _get_react_header_component(self) -> str:
        return '''import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <header className="header">
      <nav className="navbar">
        <div className="nav-container">
          <Link to="/" className="nav-brand">
            {{project_name}}
          </Link>
          <ul className="nav-menu">
            <li className="nav-item">
              <Link to="/" className="nav-link">
                Home
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/about" className="nav-link">
                About
              </Link>
            </li>
          </ul>
        </div>
      </nav>
    </header>
  );
};

export default Header;
'''

    def _get_react_footer_component(self) -> str:
        return '''import React from 'react';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <p>&copy; {{year}} {{project_name}}. All rights reserved.</p>
        <p>Built with React</p>
      </div>
    </footer>
  );
};

export default Footer;
'''

    def _get_react_home_page(self) -> str:
        return '''import React, { useState, useEffect } from 'react';

const Home = () => {
  const [message, setMessage] = useState('');

  useEffect(() => {
    setMessage('Welcome to {{project_name}}!');
  }, []);

  return (
    <div className="page">
      <div className="hero">
        <h1>{message}</h1>
        <p>This is a React application created from template.</p>
        <button className="cta-button">
          Get Started
        </button>
      </div>
    </div>
  );
};

export default Home;
'''

    def _get_react_about_page(self) -> str:
        return '''import React from 'react';

const About = () => {
  return (
    <div className="page">
      <div className="content">
        <h1>About {{project_name}}</h1>
        <p>
          This is the about page for {{project_name}}, created on {{date}}.
        </p>
        <p>
          Built with React and modern web technologies.
        </p>
      </div>
    </div>
  );
};

export default About;
'''

    def _get_react_css_template(self) -> str:
        return '''/* {{project_name}} Styles */

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  line-height: 1.6;
  color: #333;
}

.App {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header */
.header {
  background-color: #282c34;
  color: white;
  padding: 1rem 0;
}

.navbar {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.nav-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.nav-brand {
  font-size: 1.5rem;
  font-weight: bold;
  color: white;
  text-decoration: none;
}

.nav-menu {
  display: flex;
  list-style: none;
  gap: 2rem;
}

.nav-link {
  color: white;
  text-decoration: none;
  transition: color 0.3s;
}

.nav-link:hover {
  color: #61dafb;
}

/* Main Content */
.main-content {
  flex: 1;
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  width: 100%;
}

.page {
  min-height: 400px;
}

.hero {
  text-align: center;
  padding: 4rem 0;
}

.hero h1 {
  font-size: 3rem;
  margin-bottom: 1rem;
  color: #282c34;
}

.hero p {
  font-size: 1.2rem;
  margin-bottom: 2rem;
  color: #666;
}

.cta-button {
  background-color: #61dafb;
  color: #282c34;
  border: none;
  padding: 1rem 2rem;
  font-size: 1.1rem;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.cta-button:hover {
  background-color: #21a9c7;
}

.content {
  max-width: 800px;
  margin: 0 auto;
}

.content h1 {
  margin-bottom: 1rem;
  color: #282c34;
}

.content p {
  margin-bottom: 1rem;
  line-height: 1.8;
}

/* Footer */
.footer {
  background-color: #f8f9fa;
  padding: 2rem 0;
  margin-top: auto;
  text-align: center;
  border-top: 1px solid #e9ecef;
}

.footer-content p {
  margin: 0.5rem 0;
  color: #666;
}

/* Responsive */
@media (max-width: 768px) {
  .nav-container {
    flex-direction: column;
    gap: 1rem;
  }
  
  .nav-menu {
    gap: 1rem;
  }
  
  .hero h1 {
    font-size: 2rem;
  }
  
  .main-content {
    padding: 1rem;
  }
}
'''

    def _get_react_html_template(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="{{description}}" />
    <title>{{project_name}}</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
'''

    def _get_react_readme_template(self) -> str:
        return '''# {{project_name}}

A React application created on {{date}}.

## Description

{{description}}

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

## Usage

Run the development server:
```bash
npm start
```

The application will be available at http://localhost:3000

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm test` - Launches the test runner
- `npm run build` - Builds the app for production
- `npm run eject` - Ejects from Create React App

## Features

- React 18 with Hooks
- React Router for navigation
- Responsive design
- Modern CSS styling

## Author

{{author}} ({{email}})

## License

MIT License
'''

    # Express.js templates and other framework templates would follow similar patterns...
    # For brevity, I'll include placeholders for the remaining methods
    
    def _get_express_package_json(self) -> str:
        return '''{"name": "{{project_name}}", "version": "{{version}}", "main": "server.js"}'''
    
    def _get_express_server_template(self) -> str:
        return '''const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
    res.json({message: 'Welcome to {{project_name}}!'});
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});'''
    
    # Placeholder methods for other templates
    def _get_express_routes_template(self) -> str: return "// Express routes"
    def _get_express_auth_middleware(self) -> str: return "// Auth middleware"
    def _get_express_user_model(self) -> str: return "// User model"
    def _get_express_db_config(self) -> str: return "// Database config"
    def _get_express_test_template(self) -> str: return "// Express tests"
    def _get_express_readme_template(self) -> str: return "# {{project_name}}"
    def _get_express_env_template(self) -> str: return "PORT=3000"
    
    def _get_vue_package_json(self) -> str: return '''{"name": "{{project_name}}"}'''
    def _get_vue_main_template(self) -> str: return "// Vue main"
    def _get_vue_app_template(self) -> str: return "<!-- Vue App -->"
    def _get_vue_component_template(self) -> str: return "<!-- Vue Component -->"
    def _get_vue_home_view(self) -> str: return "<!-- Vue Home -->"
    def _get_vue_about_view(self) -> str: return "<!-- Vue About -->"
    def _get_vue_router_template(self) -> str: return "// Vue Router"
    def _get_vue_store_template(self) -> str: return "// Vue Store"
    def _get_vue_html_template(self) -> str: return "<!DOCTYPE html><html><head><title>{{project_name}}</title></head><body><div id=\"app\"></div></body></html>"
    def _get_vue_readme_template(self) -> str: return "# {{project_name}}"
    
    def _get_django_manage_template(self) -> str: return "#!/usr/bin/env python"
    def _get_django_requirements(self) -> str: return "Django==4.2.7"
    def _get_django_settings_template(self) -> str: return "# Django settings"
    def _get_django_urls_template(self) -> str: return "# Django URLs"
    def _get_django_wsgi_template(self) -> str: return "# Django WSGI"
    def _get_django_models_template(self) -> str: return "# Django models"
    def _get_django_views_template(self) -> str: return "# Django views"
    def _get_django_app_urls_template(self) -> str: return "# Django app URLs"
    def _get_django_base_template(self) -> str: return "<!DOCTYPE html><html><head><title>{{project_name}}</title></head><body>{% block content %}{% endblock %}</body></html>"
    def _get_django_index_template(self) -> str: return "{% extends 'base.html' %}"
    def _get_django_css_template(self) -> str: return "/* Django styles */"
    def _get_django_readme_template(self) -> str: return "# {{project_name}}"