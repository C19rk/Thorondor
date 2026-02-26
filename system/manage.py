#!/usr/bin/env python3
"""
manage.py — Argus User Management CLI
======================================
Run from the project root (same folder as app.py / wcapp.py).

Commands:
  list                                    Show all users
  make-admin   <username>                 Promote a user to admin
  revoke-admin <username>                 Remove admin from a user
  create       <username> <email> <pw>    Create a new user
                         [--admin]        Add --admin to make them an admin
  delete       <username>                 Delete a user (asks for confirmation)
  whitelist                               Show the admin whitelist
  whitelist add    <username_or_email>    Add an entry to the whitelist
  whitelist remove <username_or_email>    Remove an entry from the whitelist

Admin Whitelist
---------------
The whitelist (admin_whitelist.txt) lets you pre-approve usernames or
email addresses. When a whitelisted account registers via /signup it is
automatically granted admin status, even if other users already exist.

One entry per line. Lines starting with # are comments.
This does NOT affect existing accounts -- use make-admin for those.
"""

import sys
import os

# Make core/ importable from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Silence [AUTH] Database ready. during imports
import builtins
_orig_print = builtins.print
def _quiet_print(*a, **kw):
    if a and isinstance(a[0], str) and a[0].startswith("[AUTH]"):
        return
    _orig_print(*a, **kw)
builtins.print = _quiet_print

from core.auth import (
    init_db, get_all_users, create_user, delete_user,
    set_admin, get_connection, WHITELIST_PATH,
)

builtins.print = _orig_print  # restore normal print


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(w=62): print("─" * w)
def _ok(m):   print(f"  OK  {m}")
def _warn(m): print(f"  !   {m}")
def _info(m): print(f"      {m}")
def _err(m):
    print(f"  ERR {m}")
    sys.exit(1)


def _get_user(username: str):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, email, is_admin, created FROM users WHERE username = ?",
            (username.strip(),)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Whitelist file helpers ────────────────────────────────────────────────────

def _read_lines():
    if not os.path.exists(WHITELIST_PATH):
        return []
    with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
        return f.readlines()


def _write_lines(lines):
    # Drop blank lines, keep comments, then write with a single trailing newline
    kept = [l for l in lines if l.strip()]
    with open(WHITELIST_PATH, "w", encoding="utf-8") as f:
        for l in kept:
            f.write(l if l.endswith("\n") else l + "\n")


def _active_entries(lines):
    return {l.strip().lower() for l in lines
            if l.strip() and not l.strip().startswith("#")}


def _validate_entry(entry: str):
    if not entry:
        _err("Entry cannot be empty.")
    if " " in entry:
        _err(f"Entry cannot contain spaces: '{entry}'")
    if len(entry) > 254:
        _err("Entry is too long (max 254 chars).")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list():
    users = get_all_users()
    if not users:
        print("\n  No users in the database.\n")
        return
    print()
    _sep()
    print(f"  {'ID':<5} {'Username':<20} {'Email':<32} {'Role':<8} Created")
    _sep()
    for u in users:
        role = "ADMIN" if u["is_admin"] else "user"
        print(f"  {u['id']:<5} {u['username']:<20} {u['email']:<32} {role:<8} {u['created']}")
    _sep()
    print(f"  {len(users)} user(s) total\n")


def cmd_make_admin(username):
    u = _get_user(username)
    if not u:
        _err(f"User '{username}' not found.")
    if u["is_admin"]:
        _warn(f"'{username}' is already an admin.")
        return
    res = set_admin(u["id"], True)
    _ok(f"'{username}' is now an admin.") if res["success"] else _err(res["error"])


def cmd_revoke_admin(username):
    u = _get_user(username)
    if not u:
        _err(f"User '{username}' not found.")
    if not u["is_admin"]:
        _warn(f"'{username}' is not an admin — nothing to do.")
        return
    res = set_admin(u["id"], False)
    _ok(f"Admin removed from '{username}'.") if res["success"] else _err(res["error"])


def cmd_create(username, email, password, is_admin=False):
    res = create_user(username, email, password, is_admin=is_admin)
    if res["success"]:
        role = "admin" if is_admin else "user"
        _ok(f"User '{username}' created as {role}.")
    else:
        _err(res["error"])


def cmd_delete(username):
    u = _get_user(username)
    if not u:
        _err(f"User '{username}' not found.")
    try:
        ans = input(f"  Delete '{username}' permanently? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return
    if ans != "y":
        print("  Cancelled.")
        return
    res = delete_user(u["id"])
    _ok(f"User '{username}' deleted.") if res["success"] else _err(res["error"])


def cmd_whitelist(args):
    sub = args[0].lower() if args else "list"

    # ── list ──
    if sub in ("list", "show"):
        lines   = _read_lines()
        entries = _active_entries(lines)
        print()
        if not entries:
            _info("Whitelist is empty.")
            if not os.path.exists(WHITELIST_PATH):
                _info(f"(File not yet created: {WHITELIST_PATH})")
        else:
            _sep()
            for e in sorted(entries):
                print(f"  {e}")
            _sep()
            _info(f"{len(entries)} entry/entries")
        print()
        return

    # ── add / remove ──
    if sub not in ("add", "remove"):
        _err(f"Unknown whitelist sub-command '{args[0]}'. Use: add | remove | list")

    if len(args) < 2:
        _err(f"Usage: manage.py whitelist {sub} <username_or_email>")

    entry = args[1].strip().lower()
    _validate_entry(entry)

    lines   = _read_lines()
    entries = _active_entries(lines)

    if sub == "add":
        if entry in entries:
            _warn(f"'{entry}' is already in the whitelist.")
            return
        lines.append(entry + "\n")
        _write_lines(lines)
        _ok(f"Added '{entry}' to the whitelist.")
        _warn("Only affects future sign-ups, not existing accounts.")
        _info("To promote an existing user:  manage.py make-admin <username>")

    elif sub == "remove":
        if entry not in entries:
            _warn(f"'{entry}' is not in the whitelist — nothing to remove.")
            return
        new_lines = [l for l in lines if l.strip().lower() != entry]
        _write_lines(new_lines)
        _ok(f"Removed '{entry}' from the whitelist.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    builtins.print = _quiet_print   # silence [AUTH] during init
    init_db()
    builtins.print = _orig_print    # restore before any user-facing output
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return

    cmd = args[0].lower()

    if cmd == "list":
        cmd_list()
    elif cmd in ("make-admin", "make_admin"):
        if len(args) < 2: _err("Usage: manage.py make-admin <username>")
        cmd_make_admin(args[1])
    elif cmd in ("revoke-admin", "revoke_admin"):
        if len(args) < 2: _err("Usage: manage.py revoke-admin <username>")
        cmd_revoke_admin(args[1])
    elif cmd == "create":
        if len(args) < 4: _err("Usage: manage.py create <username> <email> <password> [--admin]")
        cmd_create(args[1], args[2], args[3], is_admin="--admin" in args)
    elif cmd == "delete":
        if len(args) < 2: _err("Usage: manage.py delete <username>")
        cmd_delete(args[1])
    elif cmd == "whitelist":
        cmd_whitelist(args[1:])
    else:
        _err(f"Unknown command '{cmd}'. Run 'python manage.py --help' for usage.")


if __name__ == "__main__":
    main()