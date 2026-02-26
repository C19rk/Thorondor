let progressInterval = null;

// ─── Recording ───────────────────────────────────────────────────────────────
function doRec(action) {
  let isStart = action === "start";
  fetch("/" + action + "_record")
    .then((r) => r.json())
    .then(() => {
      document.getElementById("startB").style.display = isStart
        ? "none"
        : "block";
      document.getElementById("stopB").style.display = isStart
        ? "block"
        : "none";
      document.getElementById("status").style.display = isStart
        ? "inline"
        : "none";
      if (!isStart) handleProgress();
      else if (progressInterval) {
        clearInterval(progressInterval);
        toggleProgressUI(false);
      }
    });
}

function handleProgress() {
  document.getElementById("startB").disabled = true;
  toggleProgressUI(true);
  const inner = document.getElementById("progressBar");
  const text = document.getElementById("progressText");

  progressInterval = setInterval(() => {
    fetch("/record_progress")
      .then((r) => r.json())
      .then((p) => {
        inner.style.width = p.percent + "%";
        text.textContent = p.percent + "%";
        if (p.done) {
          clearInterval(progressInterval);
          toggleProgressUI(false);
          document.getElementById("startB").disabled = false;
          const files =
            p.saved_files && p.saved_files.length
              ? p.saved_files
              : p.file && p.file !== "None"
                ? [p.file]
                : [];
          if (files.length) {
            alert(
              "Videos Saved!\n\n" +
                files.map((f, i) => `${i + 1}. ${f}`).join("\n"),
            );
          } else {
            alert("Videos Saved!");
          }
        }
      });
  }, 500);
}

function toggleProgressUI(show) {
  const d = show ? "block" : "none";
  document.getElementById("progress").style.display = d;
  document.getElementById("progressText").style.display = d;
  document.getElementById("saveWarning").style.display = d;
}

// ─── Log recording ───────────────────────────────────────────────────────────
function doLogRec(action) {
  let isStart = action === "start";
  fetch("/" + action + "_log_record")
    .then((r) => r.json())
    .then(() => {
      document.getElementById("startLogB").style.display = isStart
        ? "none"
        : "block";
      document.getElementById("stopLogB").style.display = isStart
        ? "block"
        : "none";
      const logStatus = document.getElementById("logStatus");
      if (isStart) {
        logStatus.style.display = "inline";
        logStatus.style.color = "#ef4444";
        logStatus.style.fontWeight = "bold";
        logStatus.style.animation = "blinker 1s linear infinite";
      } else {
        logStatus.style.display = "none";
        pollLogSaved();
      }
    });
}

function pollLogSaved() {
  const interval = setInterval(() => {
    fetch("/log_record_status")
      .then((r) => r.json())
      .then((s) => {
        if (s.saved) {
          clearInterval(interval);
          alert("Log Saved: " + s.file);
        }
      });
  }, 500);
}

// ─── Live log stream ─────────────────────────────────────────────────────────
function initLogStream() {
  const logDiv = document.getElementById("log");
  const MAX_LINES = 200;
  const evtSource = new EventSource("/log_stream");

  let pending = [];
  let rafId = null;

  function flushPending() {
    rafId = null;
    if (!pending.length) return;
    const frag = document.createDocumentFragment();
    for (const { text, color, bold } of pending) {
      const line = document.createElement("div");
      line.textContent = text;
      if (color) line.style.color = color;
      if (bold) line.style.fontWeight = "600";
      frag.appendChild(line);
    }
    pending = [];
    logDiv.appendChild(frag);
    while (logDiv.children.length > MAX_LINES)
      logDiv.removeChild(logDiv.firstChild);
    logDiv.scrollTop = logDiv.scrollHeight;
  }

  evtSource.onmessage = function (e) {
    const text = e.data;
    let color = null,
      bold = false;
    if (text.includes("Cheating")) {
      color = "#dc2626";
      bold = true;
    } else if (text.includes("Normal")) {
      color = "#16a34a";
    } else if (text.includes("Object")) {
      color = "#f97316";
    } else if (text.includes("Desk")) {
      color = "#2563eb";
    }
    pending.push({ text, color, bold });
    if (!rafId) rafId = requestAnimationFrame(flushPending);
  };
}

// ─── Logout / navigation guard ───────────────────────────────────────────────
// If a recording is active when the user tries to logout or view database,
// show a modal with two choices:
//   "Stop Recording & Logout"    → stops recorders, waits, then navigates
//   "Logout & Cancel Recordings" → navigates immediately; server stops recorders
//   "Cancel"                     → stays on page

function _buildModal(message, onStopAndGo, onCancelAndGo, onCancel) {
  const existing = document.getElementById("_logoutModal");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.id = "_logoutModal";
  overlay.style.cssText = `
    position:fixed;inset:0;background:rgba(0,0,0,.55);
    display:flex;align-items:center;justify-content:center;z-index:9999;
  `;

  overlay.innerHTML = `
    <div style="background:#fff;border-radius:14px;padding:36px 32px;
                max-width:420px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,.12);
                font-family:'Segoe UI',Inter,sans-serif;text-align:center;
                border:1px solid #e5e7eb;">
      <div style="font-size:2rem;margin-bottom:12px;">⏺️</div>
      <h2 style="margin:0 0 8px;font-size:1.1rem;font-weight:800;letter-spacing:2px;
                 color:#111827;text-transform:uppercase;">Recording in Progress</h2>
      <p style="color:#6b7280;margin:0 0 24px;line-height:1.6;font-size:13px;">${message}</p>
      <div style="display:flex;flex-direction:column;gap:8px;">
        <button id="_btnStopGo" style="
          cursor:pointer;padding:8px 14px;font-weight:600;font-size:13px;
          border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.08);
          background:#fef2f2;color:#dc2626;border:1px solid #fecaca;width:100%;">
          Stop Recording &amp; Logout
        </button>
        <button id="_btnCancelGo" style="
          cursor:pointer;padding:8px 14px;font-weight:600;font-size:13px;
          border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.08);
          background:#f9fafb;color:#4b5563;border:1px solid #4b5563;width:100%;">
          Logout &amp; Cancel Recordings
        </button>
        <button id="_btnCancel" style="
          cursor:pointer;padding:8px 14px;font-weight:600;font-size:13px;
          border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.08);
          background:#f3f4f6;color:#374151;border:1px solid #d1d5db;width:100%;">
          Cancel
        </button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);
  document.getElementById("_btnStopGo").onclick = () => {
    overlay.remove();
    onStopAndGo();
  };
  document.getElementById("_btnCancelGo").onclick = () => {
    overlay.remove();
    onCancelAndGo();
  };
  document.getElementById("_btnCancel").onclick = () => {
    overlay.remove();
    if (onCancel) onCancel();
  };
}

async function _checkAndNavigate(targetUrl) {
  let status;
  try {
    const res = await fetch("/logout_check");
    status = await res.json();
  } catch (_) {
    window.location.href = targetUrl;
    return;
  }

  const videoRec = status.recording;
  const logRec = status.log_recording;

  if (!videoRec && !logRec) {
    window.location.href = targetUrl;
    return;
  }

  // Build a precise message describing exactly what is recording
  let what;
  if (videoRec && logRec) {
    what = "A video recording and a log recording are currently in progress.";
  } else if (videoRec) {
    what = "A video recording is currently in progress.";
  } else {
    what = "A log recording is currently in progress.";
  }
  const msg = `${what} What would you like to do?`;

  _buildModal(
    msg,
    // Stop recorders cleanly, then navigate
    async () => {
      try {
        if (videoRec) await fetch("/stop_record");
        if (logRec) await fetch("/stop_log_record");
        await new Promise((r) => setTimeout(r, 800));
      } catch (_) {}
      window.location.href = targetUrl;
    },
    // Cancel recordings immediately and navigate
    async () => {
      try {
        if (videoRec) await fetch("/stop_record");
        if (logRec) await fetch("/stop_log_record");
      } catch (_) {}
      window.location.href = targetUrl;
    },
    // Stay on page
    null,
  );
}

// Intercept LOGOUT and VIEW DATABASE clicks
function _interceptNavLinks() {
  document.querySelectorAll("a.logout-btn, a[href='/logout']").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      _checkAndNavigate("/logout");
    });
  });
  document.querySelectorAll("a.admin-btn, a[href='/admin']").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      _checkAndNavigate(el.getAttribute("href") || "/admin");
    });
  });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initLogStream();
  _interceptNavLinks();
  document.getElementById("startB").disabled = false;
  document.getElementById("startLogB").disabled = false;
});
