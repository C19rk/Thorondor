"""
core/auth.py
------------
Handles all authentication logic using Python's built-in sqlite3.
NO extra packages needed — sqlite3, hashlib, and secrets are all built into Python.
"""

import sqlite3
import hashlib
import secrets
import os

# Database file lives in the project root (same folder as app.py / wcapp.py)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the users table if it doesn't exist. Called once at startup."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER  PRIMARY KEY AUTOINCREMENT,
                username TEXT     NOT NULL UNIQUE,
                email    TEXT     NOT NULL UNIQUE,
                password TEXT     NOT NULL,
                salt     TEXT     NOT NULL,
                is_admin INTEGER  NOT NULL DEFAULT 0,
                created  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Add is_admin column to existing databases that were created before this update
        try:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass  # Column already exists — that's fine
        conn.commit()
        print("[AUTH] Database ready.")
    finally:
        conn.close()


def _hash_password(password: str, salt: str) -> str:
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260000
    )
    return key.hex()


WHITELIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "admin_whitelist.txt")


def load_admin_whitelist() -> set:
    """Return a set of lowercase usernames/emails from admin_whitelist.txt."""
    if not os.path.exists(WHITELIST_PATH):
        return set()
    with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


def create_user(username: str, email: str, password: str, is_admin: bool = False) -> dict:
    """Register a new user. Returns {"success": True} or {"success": False, "error": "..."}
    The very first user ever registered is automatically made an admin.
    Users whose username or email is in admin_whitelist.txt are also made admins."""
    username = username.strip()
    email    = email.strip().lower()

    if not username or not email or not password:
        return {"success": False, "error": "All fields are required."}
    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    salt   = secrets.token_hex(32)
    hashed = _hash_password(password, salt)

    conn = get_connection()
    try:
        # If no users exist yet, first account automatically becomes admin
        existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if existing == 0:
            is_admin = True

        # Check admin whitelist (username or email match → admin)
        if not is_admin:
            whitelist = load_admin_whitelist()
            if username.lower() in whitelist or email in whitelist:
                is_admin = True

        conn.execute(
            "INSERT INTO users (username, email, password, salt, is_admin) VALUES (?, ?, ?, ?, ?)",
            (username, email, hashed, salt, 1 if is_admin else 0)
        )
        conn.commit()
        return {"success": True}
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return {"success": False, "error": "Username already taken."}
        if "email" in str(e):
            return {"success": False, "error": "Email already registered."}
        return {"success": False, "error": "Registration failed."}
    finally:
        conn.close()


def verify_user(username: str, password: str) -> dict:
    """Check login credentials. Returns {"success": True, "user": {...}} or error."""
    username = username.strip()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return {"success": False, "error": "Invalid username or password."}

        hashed = _hash_password(password, row["salt"])
        if not secrets.compare_digest(hashed, row["password"]):
            return {"success": False, "error": "Invalid username or password."}

        return {
            "success": True,
            "user": {
                "id":       row["id"],
                "username": row["username"],
                "email":    row["email"],
                "is_admin": bool(row["is_admin"]),
            }
        }
    finally:
        conn.close()


def get_all_users() -> list:
    """Return all users as a list of dicts (no passwords)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, username, email, is_admin, created FROM users ORDER BY id ASC"
        ).fetchall()
        return [
            {
                "id":       row["id"],
                "username": row["username"],
                "email":    row["email"],
                "is_admin": bool(row["is_admin"]),
                "created":  row["created"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def delete_user(user_id: int) -> dict:
    """Delete a user by ID."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def set_admin(user_id: int, is_admin: bool) -> dict:
    """Grant or revoke admin status for a user."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET is_admin = ? WHERE id = ?",
            (1 if is_admin else 0, user_id)
        )
        conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, email, is_admin FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id":       row["id"],
            "username": row["username"],
            "email":    row["email"],
            "is_admin": bool(row["is_admin"]),
        }
    finally:
        conn.close()