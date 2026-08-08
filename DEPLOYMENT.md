# Deployment Guide

PathEx can be deployed two ways, depending on who's using it and what data they're touching.

| | Streamlit Community Cloud | Docker (self-hosted) |
|---|---|---|
| Best for | Public demo, trying the app, sharing with collaborators | Institutional use, real patient data |
| Cost | Free | Your own infrastructure |
| Setup effort | ~5 minutes, no server needed | You manage a server/host |
| Storage | **Ephemeral** — wiped on redeploy/restart | Persistent, if you mount volumes |
| API key | Can use a small shared demo key | Bring your institution's own key |

If you're only trying PathEx out or sharing it publicly, use Streamlit Community Cloud. If you're processing real patient data, self-host with Docker inside your own infrastructure — see [Data & privacy](README.md#data--privacy) in the README for why.

---

## 1. Streamlit Community Cloud

1. Fork this repo (or use it directly if you have push access), so Streamlit Cloud has a repo to build from.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"Create app"** → **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository**: `<your-username>/pathex-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Before clicking Deploy, open **"Advanced settings"** and paste your secrets in TOML format:
   ```toml
   SHARED_API_KEY = "sk-..."
   SHARED_KEY_MAX_CALLS = "500"
   SHARED_KEY_MODEL = "gpt-4o-mini"
   SHARED_KEY_URL = "https://api.openai.com/v1/chat/completions"
   ```
   These populate as environment variables the same way a local `.env` does — no code changes needed. The shared key is optional; if you omit it, every visitor must bring their own key via "My own key" in the sidebar.
6. Click **Deploy**. Your app will build and go live at `https://<app-name>.streamlit.app`.

**To update secrets later**: open the app on Streamlit Cloud → **⋮ (three-dot menu)** → **Settings** → **Secrets**.

**Storage caveat**: `storage/` (user accounts) and `app_cache/` (extraction cache) live on Streamlit Cloud's ephemeral filesystem. They survive while the app is running, but are wiped on every redeploy and on periodic container restarts. Fine for a public demo; don't rely on it for long-term account/data persistence.

---

## 2. Docker (self-hosted)

Use this for a production or institutional deployment — it keeps report data inside your own infrastructure and gives you persistent storage.

### Prerequisites
- Docker installed and running ([docker.com/get-started](https://www.docker.com/get-started))
- Your own LLM API key (OpenAI, Azure OpenAI, Google Gemini, or OpenRouter) — don't reuse the publicly shared demo key for institutional traffic

### Steps

```bash
git clone https://github.com/zhuangyh/pathex-app.git
cd pathex-app

# Configure your own key — do NOT use the public demo's shared key here
cp .env.example .env
# edit .env: SHARED_API_KEY=<your institution's key>, or leave blank and require "My own key"

docker build -t pathex .

docker run -d \
  --name pathex \
  -p 8501:8501 \
  --env-file .env \
  -v pathex_storage:/app/storage \
  -v pathex_cache:/app/app_cache \
  pathex
```

Open `http://localhost:8501` (or your server's hostname/IP) once it's up.

- `-v pathex_storage:/app/storage` and `-v pathex_cache:/app/app_cache` are named Docker volumes — they persist user accounts and the extraction cache across container restarts, updates, and `docker run` re-creates. Without them, both reset every time the container restarts (same as the Streamlit Cloud caveat above).
- `--env-file .env` passes your secrets in at container start; they're never baked into the image (see `.dockerignore`).
- Check container health: `docker inspect --format='{{.State.Health.Status}}' pathex` should report `healthy` within about a minute of starting.

### Putting it behind HTTPS

Streamlit's built-in server doesn't do TLS. For a real deployment, put a reverse proxy (nginx, Caddy, or your institution's existing load balancer) in front of the container and terminate HTTPS there, forwarding to port 8501.

### Updating

```bash
git pull
docker build -t pathex .
docker stop pathex && docker rm pathex
docker run -d --name pathex -p 8501:8501 --env-file .env \
  -v pathex_storage:/app/storage -v pathex_cache:/app/app_cache pathex
```
The named volumes aren't touched by `docker rm`, so accounts and cache survive the update.

---

## 3. Running locally without Docker

For development, or just trying the app without containers:

```bash
git clone https://github.com/zhuangyh/pathex-app.git
cd pathex-app
pip install -r requirements.txt
cp .env.example .env   # optional — fill in a key, or skip and use "My own key" in the app
streamlit run app.py
```

---

## Setting up accounts

### Creating the first admin account

`users.yaml` starts empty on a fresh deployment, and the very first account ever registered is promoted to admin automatically. Just open the app and use **Create account** — no file editing needed. Every account registered after that first one gets the normal `user` role, so this only matters once, right after deploying.

To create an *additional* admin later (rather than the first one), an existing admin has to do it by hand — there's no in-app "promote to admin" control: open `users.yaml` on the server (inside the Docker volume for a self-hosted deploy, or the working directory for a local/Streamlit Cloud install) and change that account's `role: user` to `role: admin`, then have them sign out and back in.

### Resetting a user's password

Once you have at least one admin account, resets don't require touching any files:

1. Sign in as an admin → expand **⚙️ User management** in the sidebar.
2. Under **Add / update user**, enter the existing username and a new password (display name and role can be left as-is), then click **Save user**. The same form handles both creating new accounts and updating existing ones — an existing username just gets overwritten.
3. Tell the user their new password out of band — there's no email or notification system — and note that this invalidates any "remember my API key" secret they had saved, since that's encrypted with a key derived from their old password.

If you ever lose access to the *only* admin account, there's no in-app recovery. You'd need direct server/file access to hand-edit `users.yaml` with a new bcrypt hash for that account — the same fallback described above for creating the first admin.

---

## Post-deployment checklist

- [ ] Confirm the PHI pre-pass warning banner appears when uploading a reports file
- [ ] Sign in / create an account and confirm it persists after a restart (if using Docker volumes or testing Streamlit Cloud mid-session)
- [ ] If using a shared key, confirm `SHARED_KEY_MAX_CALLS` is set to a value you're comfortable funding
- [ ] Try "My own key" mode to confirm the app works without any shared key configured
- [ ] Read the in-app **Help** tab and `user_guide.html` for feature-level documentation
