"""Lightweight username/password authentication for the Streamlit app.

Users are stored in users.yaml:
  users:
    alice:
      name: Alice Smith
      password: <bcrypt hash>
      role: user          # or 'admin'

Passwords are bcrypt-hashed. Admins can add/reset users via the in-app panel.
The first account ever registered (i.e. users.yaml starts empty) is promoted
to admin automatically, so a fresh deployment doesn't need any file editing
to get its first admin -- see _do_register().
"""

import base64
import hashlib
import json
import os
from pathlib import Path

import bcrypt
import streamlit as st
import yaml
from cryptography.fernet import Fernet

USERS_FILE  = Path(__file__).parent / "users.yaml"
STORAGE_DIR = Path(__file__).parent / "storage"


# ── User file helpers ──────────────────────────────────────────────────────────

def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE) as f:
        data = yaml.safe_load(f) or {}
    return data.get("users", {})

def _save_users(users: dict):
    USERS_FILE.write_text(yaml.dump({"users": users}, default_flow_style=False))

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Password-derived encryption for optional "remember my key" storage ─────────
# bcrypt hashes (above) are one-way and can't be used to derive a repeatable
# key, so saved API keys/tokens are encrypted separately with a Fernet key
# derived (PBKDF2) from the plaintext password, which only exists in memory
# for the moment of login. The per-user salt is not secret — it's stored
# alongside the account so the same key can be re-derived on every login —
# security rests entirely on the password. Resetting a user's password (admin
# panel) intentionally invalidates any previously-saved secret.

def _derive_key(password: str, salt: bytes) -> bytes:
    raw = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, dklen=32)
    return base64.urlsafe_b64encode(raw)

def _get_or_create_salt(username: str, users: dict) -> bytes:
    info = users[username]
    salt_b64 = info.get("cred_salt")
    if salt_b64:
        return base64.b64decode(salt_b64)
    salt = os.urandom(16)
    info["cred_salt"] = base64.b64encode(salt).decode()
    _save_users(users)
    return salt

def save_user_secret(username: str, data: dict):
    """Encrypt and persist a small dict (API key/token + related fields) using
    the per-session key derived from the user's password at login. No-ops if
    that key isn't available (e.g. this session never logged in properly)."""
    key = st.session_state.get("_cred_key")
    if key is None:
        return
    blob = Fernet(key).encrypt(json.dumps(data).encode())
    (user_dir(username) / "credentials.enc").write_bytes(blob)

def load_user_secret(username: str):
    """Decrypt and return the saved secret dict, or None if there isn't one
    or it can't be decrypted (e.g. the account's password was reset since)."""
    key  = st.session_state.get("_cred_key")
    path = user_dir(username) / "credentials.enc"
    if key is None or not path.exists():
        return None
    try:
        return json.loads(Fernet(key).decrypt(path.read_bytes()))
    except Exception:
        return None

def clear_user_secret(username: str):
    path = user_dir(username) / "credentials.enc"
    if path.exists():
        path.unlink()


# ── Per-user storage ──────────────────────────────────────────────────────────

def user_dir(username: str) -> Path:
    d = STORAGE_DIR / username
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_user_state(username: str, key: str, data):
    """Persist a JSON-serialisable object to the user's storage folder."""
    path = user_dir(username) / f"{key}.json"
    path.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2))

def load_user_state(username: str, key: str, default=None):
    """Load a previously saved object, or return default."""
    path = user_dir(username) / f"{key}.json"
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def save_user_csv(username: str, key: str, df):
    """Save a DataFrame as CSV to the user's folder."""
    import pandas as pd
    path = user_dir(username) / f"{key}.csv"
    df.to_csv(path, index=False)

def load_user_csv(username: str, key: str):
    """Load a CSV into a DataFrame, or return None."""
    import pandas as pd
    path = user_dir(username) / f"{key}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return None


# ── Login / Register UI ───────────────────────────────────────────────────────

def _do_login(username: str, password: str):
    users = _load_users()
    if username in users and verify_password(password, users[username]["password"]):
        salt = _get_or_create_salt(username, users)
        st.session_state.authenticated = True
        st.session_state.username      = username
        st.session_state.display_name  = users[username].get("name", username)
        st.session_state.role          = users[username].get("role", "user")
        st.session_state._cred_key     = _derive_key(password, salt)
        st.rerun()
    else:
        st.error("Incorrect username or password.")

def _do_register(username: str, display_name: str, password: str, confirm: str):
    if not username:
        st.error("Username is required.")
        return
    if not re.match(r"^[a-z0-9_]{3,32}$", username):
        st.error("Username must be 3–32 characters: lowercase letters, digits, underscores only.")
        return
    if not password:
        st.error("Password is required.")
        return
    if len(password) < 6:
        st.error("Password must be at least 6 characters.")
        return
    if password != confirm:
        st.error("Passwords do not match.")
        return
    users = _load_users()
    if username in users:
        st.error(f"Username '{username}' is already taken.")
        return
    # Bootstrap: a fresh deployment ships no users.yaml at all, and there's
    # otherwise no way to reach the admin panel to create the first admin
    # (self-registration is normally locked to role "user"). Promoting
    # whoever registers first only fires when the user file is genuinely
    # empty, so it's a one-time bootstrap moment, not a standing hole --
    # anyone registering after account #1 still gets "user" as usual.
    role = "admin" if not users else "user"
    salt = os.urandom(16)
    users[username] = {
        "name":     display_name or username,
        "password": hash_password(password),
        "role":     role,
        "cred_salt": base64.b64encode(salt).decode(),
    }
    _save_users(users)
    # Auto-login after registration
    st.session_state.authenticated = True
    st.session_state.username      = username
    st.session_state.display_name  = display_name or username
    st.session_state.role          = role
    st.session_state._cred_key     = _derive_key(password, salt)
    st.rerun()

def login_page():
    """Render the login / register page."""
    import re as _re
    global re
    re = _re

    st.markdown(
        "<h2 style='text-align:center'>🔬 Pathology Structured Extractor</h2>",
        unsafe_allow_html=True,
    )

    col = st.columns([1, 2, 1])[1]
    with col:
        tab_login, tab_register = st.tabs(["Sign in", "Create account"])

        with tab_login:
            with st.form("login_form"):
                username  = st.text_input("Username").strip().lower()
                password  = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in",
                                use_container_width=True, type="primary")
            if submitted:
                _do_login(username, password)

        with tab_register:
            with st.form("register_form"):
                new_user    = st.text_input("Username",
                                help="3–32 chars: lowercase letters, digits, underscores")
                new_name    = st.text_input("Display name *(optional)*")
                new_pw      = st.text_input("Password *(min 6 chars)*", type="password")
                new_pw2     = st.text_input("Confirm password",         type="password")
                reg_submit  = st.form_submit_button("Create account",
                                use_container_width=True, type="primary")
            if reg_submit:
                _do_register(new_user.strip().lower(), new_name.strip(),
                             new_pw, new_pw2)


def require_login():
    """Call at the top of main(). Stops rendering if not authenticated."""
    if not st.session_state.get("authenticated"):
        login_page()
        st.stop()


def _inline_auth_form(key_prefix: str):
    """Compact Sign in / Create account tabs, embeddable anywhere (sidebar, gate)."""
    import re as _re
    global re
    re = _re

    tab_login, tab_register = st.tabs(["Sign in", "Create account"])
    with tab_login:
        with st.form(f"{key_prefix}_login_form"):
            username  = st.text_input("Username", key=f"{key_prefix}_login_user").strip().lower()
            password  = st.text_input("Password", type="password", key=f"{key_prefix}_login_pw")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if submitted:
            _do_login(username, password)
    with tab_register:
        with st.form(f"{key_prefix}_register_form"):
            new_user   = st.text_input("Username", key=f"{key_prefix}_reg_user",
                            help="3–32 chars: lowercase letters, digits, underscores")
            new_name   = st.text_input("Display name *(optional)*", key=f"{key_prefix}_reg_name")
            new_pw     = st.text_input("Password *(min 6 chars)*", type="password",
                            key=f"{key_prefix}_reg_pw")
            new_pw2    = st.text_input("Confirm password", type="password",
                            key=f"{key_prefix}_reg_pw2")
            reg_submit = st.form_submit_button("Create account", use_container_width=True,
                            type="primary")
        if reg_submit:
            _do_register(new_user.strip().lower(), new_name.strip(), new_pw, new_pw2)


def sidebar_auth_widget():
    """Render sign-in status in the sidebar without blocking the rest of the app.

    Signed in: user badge + sign-out + admin panel.
    Signed out: a compact sign-in/register form the user can ignore and keep browsing.
    """
    if st.session_state.get("authenticated"):
        sidebar_user_panel()
        admin_panel()
    else:
        with st.sidebar.expander("🔑 Sign in", expanded=False):
            _inline_auth_form("sb")
        st.sidebar.caption("Sign in to upload reports, run extractions, and save results.")


def require_login_for_action(action_label: str) -> bool:
    """Gate a specific action (not the whole page).

    Returns True if the user is authenticated. Otherwise renders an inline
    sign-in/register prompt in place and returns False so the caller can skip
    the actual (API-calling / persisting) work for this run.
    """
    if st.session_state.get("authenticated"):
        return True
    st.warning(f"🔒 Sign in to {action_label}.")
    with st.expander("Sign in / Create account", expanded=True):
        _inline_auth_form(f"gate_{action_label}".replace(" ", "_"))
    return False


# ── Sidebar logout + user info ─────────────────────────────────────────────────

def sidebar_user_panel():
    """Render the user badge and logout button in the sidebar."""
    username = st.session_state.get("username", "")
    name     = st.session_state.get("display_name", username)
    role     = st.session_state.get("role", "user")
    st.sidebar.markdown(f"**👤 {name}**  \n`{username}`"
                        + (" · admin" if role == "admin" else ""))
    if st.sidebar.button("Sign out", use_container_width=True):
        # Full reset, not just the auth flags — otherwise the uploaded
        # report, extraction results, and corrections stay visible in the
        # session after signing out. Signing back in as the same user still
        # restores their saved work from disk via _load_user_session.
        # uploader_epoch is carried over (incremented) across the clear so
        # the file_uploader widgets get a new key and remount with no file
        # attached — clearing session_state alone doesn't reset their
        # client-side visual state, since a file_uploader's displayed file
        # is tied to its widget key, not to session_state contents.
        next_epoch = st.session_state.get("uploader_epoch", 0) + 1
        st.session_state.clear()
        st.session_state["uploader_epoch"] = next_epoch
        st.rerun()


# ── Admin panel (manage users) ────────────────────────────────────────────────

def admin_panel():
    """In-app user management — only shown to admins."""
    if st.session_state.get("role") != "admin":
        return
    with st.sidebar.expander("⚙️ User management", expanded=True):
        users = _load_users()
        st.markdown(f"**{len(users)} account(s)**")
        for uname, info in users.items():
            st.text(f"  {uname}  ({info.get('name','')}  ·  {info.get('role','user')})")

        st.markdown("---")
        st.markdown("**Add / update user**")
        with st.form("add_user_form"):
            new_user = st.text_input("Username (lowercase, no spaces)").strip().lower()
            new_name = st.text_input("Display name")
            new_pw   = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["user", "admin"])
            if st.form_submit_button("Save user", type="primary"):
                if new_user and new_pw:
                    users[new_user] = {
                        "name": new_name or new_user,
                        "password": hash_password(new_pw),
                        "role": new_role,
                        "cred_salt": base64.b64encode(os.urandom(16)).decode(),
                    }
                    _save_users(users)
                    # A changed password can never re-derive the old
                    # encryption key, so any previously-saved (now
                    # permanently undecryptable) secret is just dead weight.
                    clear_user_secret(new_user)
                    st.success(f"User '{new_user}' saved.")
                    st.rerun()
                else:
                    st.warning("Username and password are required.")

        st.markdown("**Remove user**")
        with st.form("remove_user_form"):
            del_user = st.selectbox("Select user to remove",
                                    [u for u in users if u != st.session_state.username])
            if st.form_submit_button("Remove", type="secondary"):
                if del_user in users:
                    del users[del_user]
                    _save_users(users)
                    st.success(f"User '{del_user}' removed.")
                    st.rerun()
