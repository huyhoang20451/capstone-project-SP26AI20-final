# Deployment Guide

## 1. Run the full stack locally with Docker

Use the compose stack from the project root:

```bash
docker compose up --build
```

This starts:

* PostgreSQL on `localhost:5432`
* Ollama on `localhost:11434`
* FastAPI on `localhost:8000`

## 2. Push the backend to a Docker host

This project is not a good fit for Vercel as a whole because the backend depends on:

* a local PostgreSQL connection
* a local Ollama runtime
* local ML model files for Whisper and PhoBERT

Deploy the backend to a container host instead, for example:

* Render
* Railway
* Fly.io
* a VM with Docker

Set these environment variables there:

* `DATABASE_URL`
* `OLLAMA_BASE_URL`
* `DEFAULT_LLM_MODEL`
* `PHOBERT_MODEL_PATH`
* `PHOBERT_MULTITASK_MODEL_PATH`
* `WHISPER_MODEL_DIR`

## 3. Use Vercel for the frontend only

The current UI is rendered by FastAPI via Jinja templates, so it needs to be separated before Vercel can host it cleanly.

Recommended split:

* backend: keep this repository's FastAPI app in Docker
* frontend: move the UI to a static frontend or Next.js app
* frontend calls the backend over HTTPS via an environment variable such as `NEXT_PUBLIC_API_BASE_URL`

If you keep the current HTML/JS, update fetch calls from relative URLs like `/chat` to an absolute backend URL.

Example:

```javascript
const API_BASE_URL = window.API_BASE_URL || 'https://your-backend.example.com';
fetch(`${API_BASE_URL}/chat`, { method: 'POST', body: formData });
```

## 4. Vercel example

If you later move the UI to a static frontend folder, Vercel can host it with a `vercel.json` like this:

```json
{
  "version": 2,
  "builds": [{ "src": "index.html", "use": "@vercel/static" }]
}
```

For this repository as-is, do not try to deploy the FastAPI backend directly to Vercel.