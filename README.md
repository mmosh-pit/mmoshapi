# MMOSH API Documentation

## Overview
The MMOSH API provides a suite of endpoints for managing vectorized document storage, real-time AI chat generation, and voice-based AI interactions. This document outlines the functionality of each endpoint, usage scenarios, and examples.

---

**App URL** : [https://mmoshapi-uodcouqmia-uc.a.run.app](https://mmoshapi-uodcouqmia-uc.a.run.app)

---

## 🔄 1. Upload Endpoint

**Endpoint:** `POST /upload`

**Description:**  
Uploads documents (PDF/DOCX via URL or raw text) and stores them as vectorized chunks in Pinecone under a namespace.

**Request Parameters:**
- `name` (string, required): Namespace for storing vectors.
- `metadata` (string, required): Tag to attach to all data.
- `urls` (array, optional): List of file URLs (PDF or DOCX).
- `text` (string, optional): Raw text to vectorize.

**Response:**
- `200 OK`: `{ "message": "Files uploaded successfully" }`
- `4xx Error`: `{ "message": "URL [url] must point to a PDF or DOCX file." }`
- `500 Error`: `{ "message": "An error occurred: [error details]" }`

---

## 📤 2. Upload Vectors

**Page:** `test_upload.html`

**Description:**  
Web interface to upload vectors by providing document URL, metadata, and namespace.

**Steps:**
1. Open `test_upload.html`.
2. Enter namespace, metadata, and document URL(s).
3. Click `Upload`.

---

## 🗑 3. Delete Vectors

**Page:** `delete_namespace.html`

**Description:**  
Delete all vectors in a specific namespace.

**Steps:**
1. Open `delete_namespace.html`.
2. Enter the namespace (e.g., `MMOSH`).
3. Click `Delete all Vectors`.

---

## 📦 4. Fetch Namespaces

**Endpoint:** `GET /fetch_namespaces`

**Description:**  
Retrieves a list of namespaces and their metadata.

**Sample Script:**
```javascript
async function fetchNamespaces() {
  const res = await fetch('https://mmoshapi-uodcouqmia-uc.a.run.app/fetch_namespaces');
  const data = await res.json();
  console.log('Namespaces:', data.namespaces);
}
```

**Response:**
- `200 OK`: List of namespaces with vector count and metadata.
- `500 Error`: Server error message.

---

## ⚡ 5. Streamed AI Response

**Endpoint:** `POST /generate_stream/`

**Description:**  
Returns AI-generated text in chunks (real-time streaming).

**Request Body:**
```json
{
  "username": "exampleUser",
  "prompt": "What is the capital of France?",
  "namespaces": ["exampleNamespace"],
  "metafield": "",
  "system_prompt": "Provide concise answers.",
  "model": "llama-3.2-1b-preview",
  "session_id": "84e7cc93-b833-46f6-9312-4815f620b89d"
}
```

**Response:**
- Streamed chunks of generated text.

---

## 💬 6. Static AI Response

**Endpoint:** `POST /generate/`

**Description:**  
Returns a full AI response based on input prompt and options.

**Request Body:**
```json
{
  "username": "unique_user",
  "prompt": "I'm looking for a new phone",
  "chat_history": [
    ["human", "Hi"],
    ["ai", "Hello, how can I help you today?"],
    ["human", "I'm looking for a new phone"]
  ],
  "namespaces": ["TESTO"],
  "metafield": "",
  "system_prompt": "Optional system prompt",
  "model": "llama-3.2-1b-preview",
  "session_id": "84e7cc93-b833-46f6-9312-4815f620b89d"
}
```

**Response:**
- `200 OK`: Plain text AI response.
- `400 Error`: Missing required fields.
- `500 Error`: Processing issue.

---

## 7. Voice AI (WebSocket)

**Endpoint** : `WebSocket /ws`

**Description:**
Streams audio from the user and provides real-time voice-based AI responses. Internally, it uses a reactive AI agent that pulls relevant context using tools and returns streamed speech responses. All interactions are logged in LangSmith with session tracing.

**Initial Config (Sent as JSON):**

```json
{
  "username": "alirehman",
  "chat_history": [],
  "metafield": "user-profile-tag",
  "system_prompt": "You are a helpful assistant.",
  "namespaces": ["deepseek3"],
  "session_id" : "84e7cc93-b833-46f6-9312-4815f620b89d"
}
```


**Headers:**

session_id (optional): A session identifier used for tracing (auto-generated if missing).

**Instruction Logic (automatically built):**

```
{system_prompt}

- Always use the tool to get context before answering.
- Use the tool **every time** the user asks something.
- Always include:
- namespaces = {namespaces}
- metafield = {metafield}

If the fetched context isn’t relevant, answer using your own knowledge.
Stop VOICE **immediately** if the user starts talking while you are answering.
```


**Audio Format:**

Input/Output is base64-encoded PCM audio at 24kHz.

Transcripts and response instructions are handled by OpenAIVoiceReactAgent.

**What’s Logged to LangSmith:**

username, session_id, model_name, and full instruction string.

Each voice session is stored as a chain run with status stream_sent.

**Response Example:**
```
On connection: { "status": "connected" }
```

Then: streamed voice/text response.
---

## 🎧 8. Audio Streaming Web Interface

**Endpoint:** `GET /audio`

**Description:**  
This endpoint serves the audio streaming web interface. It opens a webpage where users can start streaming their microphone audio and hear the AI-generated response in real-time.

**Usage:**
- **URL:** `https://mmoshapi-uodcouqmia-uc.a.run.app/audio`
- **Method:** `GET`
- **Response:** HTML page with a "Start Audio" button.

**What it Does:**
- Connects to the backend WebSocket (`/ws`)
- Captures audio from the user’s microphone
- Streams audio to the server
- Receives audio responses and plays them back to the user

**How to Test:**
1. Open the URL in your browser.
2. Click `Start Audio`.
3. Grant microphone permission.
4. Speak and hear the AI respond back.


**Endpoint:** `WebSocket /ws`

**Description:**  
Streams audio from user and sends real-time AI voice responses.

**Initial Config (JSON):**
```json
{
  "model_name": "gemini-2.0-flash",
  "username": "alirehman",
  "chat_history": [],
  "metafield": " ",
  "system_prompt": "You are a helpful assistant.",
  "namespaces": ["deepseek3"]
}
```

**Audio Stream Format:**
- Input and output are base64-encoded PCM (24kHz).

---

## 🧪 8. How to Test

**Run Locally:**
```bash
git clone https://github.com/mmosh-pit/mmoshapi
cd mmoshapi
uvicorn app:app
```

**Web Pages to Test:**
- `/audio` – Voice chat
- `chat.html` – Text chat
- `fetch_data.html` – View namespaces
- `delete_namespace.html` – Delete vectors
- `test_upload.html` – Upload vectors

---

## 📌 Notes
- Uses Pinecone for vector store.
- Embedding via VertexAI.
- Voice support through WebSocket.
