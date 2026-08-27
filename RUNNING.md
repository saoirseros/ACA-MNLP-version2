# How to Run This Project (Step-by-Step)

This is the detailed, beginner-friendly companion to the "Getting Started"
section in the root [README.md](README.md). It walks through everything
needed to run the full system locally on Windows: the MongoDB database,
the Node/Express + Socket.io backend, the Python FastAPI NLP service, and
the React/Vite frontend.

If you just want the short version, see [README.md → Getting
Started](README.md#getting-started). This file is the long version, with
explanations, troubleshooting, and the optional experiment/training/testing
steps.

---

## 0. Overview: what you're running

This project is **four separate processes** that talk to each other:

| # | Process | Technology | Default URL/Port |
| - | --- | --- | --- |
| 1 | Database | MongoDB | `mongodb://localhost:27017` |
| 2 | Backend (chat API + Socket.io) | Node.js / Express | `http://localhost:5000` |
| 3 | NLP service (sentiment/emotion/toxicity/summary/topic + Adaptive Context Activation) | Python / FastAPI | `http://localhost:8000` |
| 4 | Frontend (chat UI) | React / Vite | `http://localhost:5173` |

They must be started **in this order** (1 → 2 → 3 → 4), because the
backend needs the database to already be reachable, and the frontend
needs the backend to already be reachable. The NLP service is optional
in the sense that the chat still works without it - if it's not running
or not configured, messages just won't get sentiment/emotion/toxicity
analysis, and the UI shows a small "NLP analysis is currently
unavailable" banner instead of breaking.

---

## 1. Prerequisites

Install these once, if you don't already have them:

- **Node.js** 18 or newer (this project was developed with Node 22). [nodejs.org](https://nodejs.org/)
- **Python** 3.10-3.12 (this project was developed with Python 3.11). [python.org](https://www.python.org/downloads/)
- **MongoDB Community Server** (for a local database), or a free
  [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) cluster
  if you'd rather not install MongoDB locally.
- **Git**, to clone the repository.
- (Optional) A free [Cloudinary](https://cloudinary.com/) account, only
  needed if you want profile pictures / image messages to actually
  upload. Everything else works without it.

None of the above requires paying for anything - MongoDB Community,
Cloudinary's free tier, and every Hugging Face model this project uses
are all free.

---

## 2. Get the code

```powershell
git clone <your-fork-or-repo-url>
cd "PROJECT PHASE 1"   # or wherever you cloned it
```

The three main folders you'll work in are `server/`, `client/`, and
`nlp-service/`.

---

## 3. Start MongoDB

If you installed MongoDB Community Server, it usually runs as a Windows
service already. To check, or to start it manually:

```powershell
# Check if it's already running
Get-Process -Name mongod -ErrorAction SilentlyContinue

# If nothing is printed, start it manually (adjust the path if your
# MongoDB install location differs):
mongod
```

If you're using MongoDB Atlas instead of a local install, you don't need
to start anything here - just make sure you have your Atlas connection
string ready for the next step.

---

## 4. Start the backend (Node/Express server)

```powershell
cd server
npm install
copy .env.example .env
```

Now open `server/.env` in an editor and fill in real values:

| Variable | What to put |
| --- | --- |
| `MONGODB_URI` | `mongodb://localhost:27017` for a local MongoDB, or your Atlas connection string (without a trailing database name - `/chat-app` is appended automatically) |
| `JWT_SECRET` | Any long random string, e.g. generate one with `node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"` |
| `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` | From your Cloudinary dashboard, if you want image uploads to work. Leave blank to skip - only image/profile-picture uploads will fail, chat text still works fine |
| `PORT` | Leave as `5000` unless that port is taken |
| `NODE_ENV` | Leave as `development` for local runs |
| `NLP_SERVICE_URL` | `http://localhost:8000` (matches step 5 below). Leave this line blank/commented out if you don't want to run the NLP service at all - chat still works, just without analysis |

Then start it:

```powershell
npm run server
```

You should see:

```
Database Connected
Server is running on PORT: 5000
```

Leave this terminal running. Quick check in a second terminal:

```powershell
curl http://localhost:5000/api/status
# -> "Server is live"
```

---

## 5. Start the NLP service (Python/FastAPI)

Open a **new terminal** for this step (keep the backend running in the
first one).

```powershell
cd nlp-service
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

> **Windows + OneDrive/long-path note:** if your project folder lives
> under a deeply nested path (a synced OneDrive folder is a common
> cause), installing `torch` above can fail with an error like
> `WinError 206: filename or extension is too long`. If that happens,
> create the virtual environment somewhere with a short path instead,
> e.g.:
> ```powershell
> python -m venv C:\nlp-venv
> C:\nlp-venv\Scripts\python.exe -m pip install -r requirements.txt
> ```
> and use `C:\nlp-venv\Scripts\python.exe` in place of `.venv\Scripts\python.exe`
> in every command below. See [nlp-service/README.md](nlp-service/README.md)
> for more detail. (In this project's own development environment, the
> venv lives at `C:\nlp-venvs\ptp-nlp-service` for exactly this reason.)

The default values in `nlp-service/.env` work out of the box - no
editing required unless you want to change the port or logging level.

Now start the service:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

(If you used the short-path venv workaround, use its `python.exe`
instead.) You should see something like:

```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
... | INFO | app.main | nlp-service v0.1.0 starting up (host=0.0.0.0, port=8000)
... | INFO | app.inference.warmup | Warming up NLP models...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**The first startup downloads several pretrained models from Hugging
Face** (roughly 1-1.5 GB total across sentiment/emotion/toxicity/
summarization/embedding models), so the "warming up" step can take a
few minutes the very first time, and needs an internet connection.
Subsequent startups are fast because the models are cached locally.

Quick check in another terminal:

```powershell
curl http://localhost:8000/health
# -> {"status":"ok","service":"nlp-service","version":"0.1.0"}
```

If this whole step feels like too much for now, you can skip it: just
leave `NLP_SERVICE_URL` unset (or comment it out) in `server/.env`, and
the chat app will run fine without message analysis.

---

## 6. Start the frontend (React/Vite client)

Open a **third terminal**.

```powershell
cd client
npm install
copy .env.example .env
npm run dev
```

`client/.env` only needs one variable, and the default is already
correct if you kept the backend on port 5000:

```
VITE_BACKEND_URL=http://localhost:5000
```

You should see Vite print a local URL, typically:

```
  VITE vX.X.X  ready in ... ms
  ➜  Local:   http://localhost:5173/
```

Open that URL in your browser.

---

## 7. Try it out

1. On the login page, create two different accounts (e.g. in one normal
   browser window and one private/incognito window, so you can be
   "logged in as" two different users at once).
2. Send a message from one account to the other.
3. It should appear instantly on both sides (this is the real-time chat
   working, independent of the NLP service).
4. A few hundred milliseconds later, a small colored badge should
   appear under the message (e.g. "positive (92%)", an emotion label,
   and a "ctx: low/medium/high" tag) - this is the NLP analysis arriving
   asynchronously. If the NLP service isn't running, you'll instead see
   an amber "NLP analysis is currently unavailable" banner at the top of
   the chat, and no badges - this is expected, working behavior, not a
   bug.
5. Once a few messages have been exchanged and analyzed, open the right
   sidebar for that conversation - you should see a "Conversation
   Intelligence" panel with overall sentiment, dominant emotion, a live
   summary, and context-usage stats.

If steps 1-3 work but 4-5 don't, double check `NLP_SERVICE_URL` in
`server/.env` and that the NLP service terminal is still running and
says "Application startup complete."

---

## 8. Running the automated tests (optional, but quick)

**Backend tests** (Node's built-in test runner, no extra install needed):

```powershell
cd server
npm test
```

Expected: `# pass 2`, `# fail 0`.

**NLP context-algorithm tests** (verifies Adaptive Context Activation
against the exact canonical examples from `nlp-service/README.md`):

```powershell
cd nlp-service
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

(Use your short-path venv's `python.exe` if you used the workaround in
step 5.) Expected: 3 tests, `OK`. The first run may take a little longer
while it loads the cached embedding/sentiment models into memory.

---

## 9. Optional: reproducing the experiment results

`experiments/` contains the code that generated the real
accuracy/F1/latency numbers written up in
[experiments/FINDINGS.md](experiments/FINDINGS.md) (baseline vs
Transformer per task, and full-context vs Adaptive Context Activation).
This is **not required** to run the chat app - it's a separate,
standalone research/evaluation step.

```powershell
cd experiments
<path-to-your-nlp-service-venv>\Scripts\python.exe -m pip install -r ..\nlp-service\requirements.txt
<path-to-your-nlp-service-venv>\Scripts\python.exe run_all.py
```

This takes roughly 10-15 minutes on a CPU-only machine (it downloads a
few public datasets the first time, then runs real model inference on
them). Results are written to `experiments/results/` as JSON files and
PNG charts. See [experiments/README.md](experiments/README.md) for full
detail.

---

## 10. Optional: the fine-tuning scaffold

`training/` contains scripts to fine-tune a base Transformer on each
task's dataset. These are **not run by default** (CPU-only fine-tuning
is slow) and are not required for anything else in this project to
work. See [training/README.md](training/README.md) if you want to try
it.

---

## Quick reference: all environment variables

**`server/.env`**

| Variable | Required? | Purpose |
| --- | --- | --- |
| `MONGODB_URI` | Yes | Database connection |
| `JWT_SECRET` | Yes | Signs login tokens |
| `CLOUDINARY_CLOUD_NAME` / `CLOUDINARY_API_KEY` / `CLOUDINARY_API_SECRET` | No | Only for image uploads |
| `PORT` | No (default 5000) | Backend port |
| `NODE_ENV` | No (default development) | Runtime mode |
| `NLP_SERVICE_URL` | No | Enables message analysis when set to the running NLP service's URL |

**`nlp-service/.env`**

| Variable | Required? | Purpose |
| --- | --- | --- |
| `NLP_SERVICE_HOST` / `NLP_SERVICE_PORT` | No (defaults shown in `.env.example`) | Bind address/port |
| `NLP_ALLOWED_ORIGINS` | No | CORS origins allowed to call this service directly |
| `NLP_LOG_LEVEL` | No | Logging verbosity |

**`client/.env`**

| Variable | Required? | Purpose |
| --- | --- | --- |
| `VITE_BACKEND_URL` | Yes | Where the frontend sends API/Socket.io requests |

---

## Troubleshooting

- **`Error: connect ECONNREFUSED ...27017` when starting the backend** -
  MongoDB isn't running, or `MONGODB_URI` is wrong. Re-check step 3.
- **`WinError 206: filename or extension is too long` installing
  `torch`** - see the long-path note in step 5; create the venv at a
  shorter path.
- **Login works but chat messages never get analyzed / no badges ever
  appear** - the NLP service isn't reachable. Check that its terminal is
  still running, `curl http://localhost:8000/health` succeeds, and
  `NLP_SERVICE_URL` in `server/.env` matches its actual URL/port. This is
  expected to degrade gracefully (chat itself keeps working) rather than
  crash anything.
- **Port already in use** - another process is already listening on
  5000/8000/5173. Either stop that process, or change the port in the
  relevant `.env` file (and update `VITE_BACKEND_URL` / `NLP_SERVICE_URL`
  to match if you change the backend or NLP service port).
- **Image/profile picture upload fails** - `CLOUDINARY_*` variables are
  empty. Either fill them in with a free Cloudinary account's
  credentials, or just don't use image uploads - text chat is unaffected.

---

## Stopping everything

Press `Ctrl+C` in each of the three (or four, if MongoDB is running in
its own terminal) terminal windows, in any order.
