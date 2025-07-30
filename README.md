# Simple AI Assistant

A basic web-based AI assistant with file upload capabilities.

## Features

- Simple chat interface
- File upload support
- Basic API key management
- Task status tracking

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys in `config/config.toml`
4. Run the application:
```bash
python app.py
```

5. Open your browser to `http://localhost:5000`

## Usage

- Type messages in the chat interface
- Upload files to include in conversations
- View API key status
- Simple task management

## API Endpoints

- `POST /api/run-flow` - Send message
- `GET /api/task-status/<task_id>` - Check task status
- `POST /api/upload` - Upload files
- `GET /api/files` - List uploaded files
- `GET /api/keys/status` - API key status

## Configuration

Add your API keys to `config/config.toml`:

```toml
[llm]
[[llm.api_keys]]
api_key = "your-api-key"
name = "API Key Name"
enabled = true
```

## License

MIT License