let progressInterval = null;

// ─── Recording ───────────────────────────────────────────────────────────────
function doRec(action) {
  const isStart = action === "start";
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
        progressInterval = null;
        toggleProgressUI(false);
      }
    });
}

function handleProgress(navigateAfter) {
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
          progressInterval = null;
          toggleProgressUI(false);
          document.getElementById("startB").disabled = false;

          // Show saved filenames
          const files =
            p.saved_files && p.saved_files.length
              ? p.saved_files
              : p.file && p.file !== "None"
                ? [p.file]
                : [];
          const msg = files.length
            ? "Videos Saved!\n\n" +
              files.map((f, i) => `${i + 1}. ${f}`).join("\n")
            : "Videos Saved!";

          if (navigateAfter) {
            alert(msg);
            window.location.href = navigateAfter;
          } else {
            alert(msg);
          }
        }
      })
      .catch(() => {});
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
  const isStart = action === "start";
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

function pollLogSaved(navigateAfter) {
  const interval = setInterval(() => {
    fetch("/log_record_status")
      .then((r) => r.json())
      .then((s) => {
        if (s.saved) {
          clearInterval(interval);
          if (navigateAfter) {
            alert("Log Saved: " + s.file);
            window.location.href = navigateAfter;
          } else {
            alert("Log Saved: " + s.file);
          }
        }
      })
      .catch(() => {});
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

// ─── Navigation guard ────────────────────────────────────────────────────────
// Shown when the user clicks Logout or View Database while recording.
// "Save and Stop Recording" — stops cleanly, shows saved filenames, stays on page.
// "Cancel"                  — dismisses the modal, keeps recording, no navigation.

function _buildModal(message, onSaveAndStop) {
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
      <h2 style="margin:0 0 8px;font-size:1.1rem;font-weight:800;letter-spacing:2px;
                 color:#111827;text-transform:uppercase;">Recording in Progress</h2>
      <p style="color:#6b7280;margin:0 0 24px;line-height:1.6;font-size:13px;">${message}</p>
      <div style="display:flex;flex-direction:column;gap:8px;">
        <button id="_btnSaveStop" style="
          cursor:pointer;padding:8px 14px;font-weight:600;font-size:13px;
          border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.08);
          background:#fef2f2;color:#dc2626;border:1px solid #fecaca;width:100%;">
          Save and Stop Recording
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
  document.getElementById("_btnSaveStop").onclick = () => {
    overlay.remove();
    onSaveAndStop();
  };
  document.getElementById("_btnCancel").onclick = () => {
    overlay.remove();
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

  // Nothing recording — navigate straight through
  if (!videoRec && !logRec) {
    window.location.href = targetUrl;
    return;
  }

  // Precise message — only mention what is actually recording
  let what;
  if (videoRec && logRec) {
    what = "A video recording and a log recording are currently in progress.";
  } else if (videoRec) {
    what = "A video recording is currently in progress.";
  } else {
    what = "A log recording is currently in progress.";
  }

  _buildModal(
    `${what} Please save and stop before proceeding.`,
    // "Save and Stop Recording" — stop cleanly, show filenames, stay on page
    async () => {
      try {
        if (videoRec) await fetch("/stop_record");
        if (logRec) await fetch("/stop_log_record");
      } catch (_) {}

      if (videoRec) {
        // Show progress bar and wait for finalization, then show saved filenames
        document.getElementById("startB").disabled = true;
        toggleProgressUI(true);
        const inner = document.getElementById("progressBar");
        const text = document.getElementById("progressText");
        const poll = setInterval(() => {
          fetch("/record_progress")
            .then((r) => r.json())
            .then((p) => {
              inner.style.width = p.percent + "%";
              text.textContent = p.percent + "%";
              if (p.done) {
                clearInterval(poll);
                toggleProgressUI(false);
                document.getElementById("startB").disabled = false;
                // Reset UI to show Record Video button again
                document.getElementById("startB").style.display = "block";
                document.getElementById("stopB").style.display = "none";
                document.getElementById("status").style.display = "none";
                const files =
                  p.saved_files && p.saved_files.length
                    ? p.saved_files
                    : p.file && p.file !== "None"
                      ? [p.file]
                      : [];
                alert(
                  files.length
                    ? "Videos Saved!\n\n" +
                        files.map((f, i) => `${i + 1}. ${f}`).join("\n")
                    : "Videos Saved!",
                );
              }
            })
            .catch(() => {
              clearInterval(poll);
              toggleProgressUI(false);
            });
        }, 500);
      }

      if (logRec) {
        // Reset log UI and wait for save confirmation
        document.getElementById("startLogB").style.display = "block";
        document.getElementById("stopLogB").style.display = "none";
        document.getElementById("logStatus").style.display = "none";
        pollLogSaved();
      }
    },
    // "Cancel" button just removes the modal (no second arg needed)
  );
}

// Intercept LOGOUT and VIEW DATABASE clicks using capture phase
// so the listener fires before the browser follows the href
function _interceptNavLinks() {
  document.addEventListener(
    "click",
    (e) => {
      const el = e.target.closest(
        "a.logout-btn, a[href='/logout'], a.admin-btn, a[href='/admin']",
      );
      if (!el) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      _checkAndNavigate(el.getAttribute("href") || "/logout");
    },
    true,
  ); // capture: true
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initLogStream();
  _interceptNavLinks();
  document.getElementById("startB").disabled = false;
  document.getElementById("startLogB").disabled = false;
});
