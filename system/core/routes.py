import os
import asyncio
from fastapi import Request, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response, RedirectResponse

from core.auth import create_user, verify_user, get_all_users, delete_user, set_admin


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wants_json(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "application/json" in accept or str(request.url.path) in (
        "/start_record", "/stop_record", "/record_progress",
        "/recorder_status", "/start_log_record", "/stop_log_record",
        "/log_record_status", "/log_stream"
    )


def _not_logged_in_response(request: Request):
    """Browser → redirect to login. API call → 401 JSON."""
    if _wants_json(request):
        return JSONResponse(
            {"status": "error", "message": "Not logged in. Please log in first."},
            status_code=401
        )
    return RedirectResponse("/login?reason=session_expired", status_code=302)


def _not_admin_response(request: Request):
    """User is logged in but not an admin."""
    if _wants_json(request):
        return JSONResponse(
            {"status": "error", "message": "Admin access required."},
            status_code=403
        )
    from fastapi.responses import HTMLResponse
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Access Denied — Argus</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:"Segoe UI",Inter,system-ui,sans-serif;min-height:100vh;
         display:flex;align-items:center;justify-content:center;
         background:linear-gradient(135deg,#f4f6f9,#e9edf3);color:#1f2937}
    .card{background:#fff;border-radius:16px;padding:48px 40px;text-align:center;
          box-shadow:0 20px 60px rgba(0,0,0,.1);max-width:420px;width:100%}
    .icon{font-size:3rem;margin-bottom:16px}
    h1{font-size:1.5rem;font-weight:800;letter-spacing:3px;color:#111827;margin-bottom:8px}
    p{color:#6b7280;margin-bottom:28px;line-height:1.6}
    .btn{display:inline-block;padding:10px 22px;border-radius:8px;font-weight:600;
         font-size:14px;text-decoration:none;transition:all .2s}
    .btn-primary{background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;
                 box-shadow:0 4px 12px rgba(59,130,246,.35)}
    .btn-ghost{background:#f3f4f6;color:#374151;margin-left:10px;border:1px solid #d1d5db}
    .btn:hover{transform:translateY(-2px)}
  </style>
</head>
<body>
  <div class="card">
    <h1>ACCESS DENIED!</h1>
    <p>You need <strong>admin</strong> privileges to view this page.<br>
       Contact an administrator or use <code>manage.py</code> to promote your account.</p>
    <a href="/" class="btn btn-primary">&larr; Back to App</a>
    <a href="/logout" class="btn btn-ghost">Logout</a>
  </div>
</body>
</html>"""
    return HTMLResponse(html, status_code=403)


def register_routes(app, recorder, log_recorder, generate_frames, frames,
                    CAMERA_SOURCES, handle_offer, LOG_FILE, follow,
                    templates=None, template_name="app.html"):

    # ── Auth: Login ───────────────────────────────────────────────────────────
    @app.get("/login")
    async def login_page(request: Request):
        if request.session.get("user_id"):
            return RedirectResponse("/", status_code=302)
        reason = request.query_params.get("reason", "")
        error  = "Your session has expired. Please log in again." if reason == "session_expired" else None
        return templates.TemplateResponse("login.html", {"request": request, "error": error})

    @app.post("/login")
    async def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...)
    ):
        result = verify_user(username, password)
        if not result["success"]:
            return templates.TemplateResponse("login.html", {
                "request":  request,
                "error":    result["error"],
                "username": username,
            })
        request.session["user_id"]  = result["user"]["id"]
        request.session["username"] = result["user"]["username"]
        request.session["is_admin"] = result["user"]["is_admin"]
        return RedirectResponse("/", status_code=302)

    # ── Auth: Signup ──────────────────────────────────────────────────────────
    @app.get("/signup")
    async def signup_page(request: Request):
        if request.session.get("user_id"):
            return RedirectResponse("/", status_code=302)
        return templates.TemplateResponse("signup.html", {"request": request})

    @app.post("/signup")
    async def signup_submit(
        request: Request,
        username: str         = Form(...),
        email: str            = Form(...),
        password: str         = Form(...),
        confirm_password: str = Form(...)
    ):
        if password != confirm_password:
            return templates.TemplateResponse("signup.html", {
                "request":  request,
                "error":    "Passwords do not match.",
                "username": username,
                "email":    email,
            })
        result = create_user(username, email, password)
        if not result["success"]:
            return templates.TemplateResponse("signup.html", {
                "request":  request,
                "error":    result["error"],
                "username": username,
                "email":    email,
            })
        # Show the success popup — user clicks "Go to Login Page" themselves
        return templates.TemplateResponse("signup.html", {
            "request":    request,
            "registered": True,
        })

    # ── Auth: Logout ──────────────────────────────────────────────────────────
    @app.get("/logout_check")
    async def logout_check(request: Request):
        """Return recording status so the frontend can warn before logout."""
        if not request.session.get("user_id"):
            return JSONResponse({"recording": False, "log_recording": False})
        return JSONResponse({
            "recording":     bool(getattr(recorder,     "recording", False)),
            "log_recording": bool(getattr(log_recorder, "recording", False)),
        })

    @app.get("/logout")
    async def logout(request: Request):
        # Recordings keep running — they are server-side background processes.
        # The JS modal asks the user to stop them first; but if they force
        # logout we still just clear the session and redirect.
        request.session.clear()
        return RedirectResponse("/login", status_code=302)

    # ── Admin Panel ───────────────────────────────────────────────────────────
    @app.get("/admin")
    async def admin_page(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        if not request.session.get("is_admin"):
            return _not_admin_response(request)

        message      = request.query_params.get("message", None)
        message_type = request.query_params.get("type", "success")

        return templates.TemplateResponse("admin.html", {
            "request":         request,
            "users":           get_all_users(),
            "username":        request.session.get("username"),
            "current_user_id": request.session.get("user_id"),
            "message":         message,
            "message_type":    message_type,
        })

    @app.post("/admin/delete/{user_id}")
    async def admin_delete_user(request: Request, user_id: int):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        if not request.session.get("is_admin"):
            return _not_admin_response(request)
        if user_id == request.session.get("user_id"):
            return RedirectResponse("/admin?message=You+cannot+delete+yourself.&type=error", status_code=302)
        result = delete_user(user_id)
        if result["success"]:
            return RedirectResponse("/admin?message=User+deleted+successfully.&type=success", status_code=302)
        return RedirectResponse(f"/admin?message=Error:+{result['error']}&type=error", status_code=302)

    @app.post("/admin/make_admin/{user_id}")
    async def admin_make_admin(request: Request, user_id: int):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        if not request.session.get("is_admin"):
            return _not_admin_response(request)
        set_admin(user_id, True)
        return RedirectResponse("/admin?message=User+promoted+to+admin.&type=success", status_code=302)

    @app.post("/admin/remove_admin/{user_id}")
    async def admin_remove_admin(request: Request, user_id: int):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        if not request.session.get("is_admin"):
            return _not_admin_response(request)
        if user_id == request.session.get("user_id"):
            return RedirectResponse("/admin?message=You+cannot+remove+your+own+admin+status.&type=error", status_code=302)
        set_admin(user_id, False)
        return RedirectResponse("/admin?message=Admin+status+removed.&type=success", status_code=302)

    # ── Main App (protected) ──────────────────────────────────────────────────
    @app.get("/", response_class=Response)
    async def index(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        default_cam = list(CAMERA_SOURCES.keys())[0]
        return templates.TemplateResponse(
            template_name,
            {
                "request":     request,
                "cams":        list(CAMERA_SOURCES.keys()),
                "default_cam": default_cam,
                "username":    request.session.get("username"),
                "is_admin":    request.session.get("is_admin", False),
            }
        )

    # ── Recording (protected) ─────────────────────────────────────────────────
    @app.get("/start_record")
    async def start_record(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        if not getattr(recorder, "directory_set", False):
            return JSONResponse({"status": "error", "message": "Please set directory first"}, status_code=400)
        cam_name = request.query_params.get("cam_name", None)
        if cam_name:
            recorder.start(cam_name=cam_name)
        else:
            recorder.start(cam_names=list(CAMERA_SOURCES.keys()))
        return JSONResponse({"status": "Started"})

    @app.get("/stop_record")
    async def stop_record(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        recorder.stop()
        return JSONResponse({"status": "Stop requested"})

    @app.get("/record_progress")
    async def record_progress(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        done = not recorder.finalizing and not recorder.recording
        return JSONResponse({
            "status":  recorder.status_msg,
            "file":    os.path.basename(recorder.current_file) if recorder.current_file else "None",
            "percent": 100 if not recorder.finalizing else 50,
            "done":    done,
            "saved_files": list(recorder.saved_files) if done else [],
        })

    @app.get("/recorder_status")
    async def recorder_status(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        return JSONResponse({
            "recording": recorder.recording,
            "status":    recorder.status_msg,
            "file":      os.path.basename(recorder.current_file) if recorder.current_file else "None",
            "path":      recorder.output_dir
        })

    # ── Log Recording (protected) ─────────────────────────────────────────────
    @app.get("/start_log_record")
    async def start_log_record(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        if not getattr(log_recorder, "directory_set", False):
            return JSONResponse({"status": "error", "message": "Please set log directory first"}, status_code=400)
        log_recorder.start()
        return JSONResponse({"status": "Started"})

    @app.get("/stop_log_record")
    async def stop_log_record(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        log_recorder.stop()
        return JSONResponse({"status": "Stop requested"})

    @app.get("/log_record_status")
    async def log_record_status(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        return JSONResponse({
            "recording":  log_recorder.recording,
            "finalizing": log_recorder.finalizing,
            "saved":      log_recorder.saved,
            "file":       os.path.basename(log_recorder.filename) if log_recorder.filename else "None"
        })

    # ── Log Stream SSE (protected) ────────────────────────────────────────────
    @app.get("/log_stream")
    async def log_stream(request: Request):
        if not request.session.get("user_id"):
            return _not_logged_in_response(request)
        if not os.path.exists(LOG_FILE):
            open(LOG_FILE, "w").close()

        async def event_generator():
            logfile = open(LOG_FILE, "r")
            logfile.seek(0, 2)
            try:
                while True:
                    line = logfile.readline()
                    if not line:
                        await asyncio.sleep(0.1)
                        continue
                    if log_recorder.recording:
                        log_recorder.write(line.strip())
                    yield f"data: {line}\n\n"
            finally:
                logfile.close()

        return StreamingResponse(event_generator(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})