# Day 03 — HTTP & APIs 🌐

## 🎯 Objective
Understand how applications communicate with external services and AI models through APIs.

## 📚 Concepts Covered
- HTTP fundamentals
- GET requests
- POST requests
- Headers
- JSON
- REST APIs
- API keys
- Environment variables
- Direct AI API integration

## 🛠️ Build
### Direct AI API Call
Implemented an AI API integration **without LangChain** to understand what happens underneath higher-level frameworks.

Basic flow:

```text
Python Application
       ↓
HTTP Request
       ↓
AI API
       ↓
JSON Response
       ↓
Python Application
```

## 🔐 Security
API credentials should be stored in environment variables rather than hard-coded into source code.

## 🧠 Key Takeaways
Understanding HTTP and APIs is essential before using frameworks such as LangChain or agent frameworks.
