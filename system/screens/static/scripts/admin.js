// admin.js

// ── Confirmation dialogs ──────────────────────────────────────────────────────

function confirmDelete(username) {
  return confirm('Delete user "' + username + '"?\n\nThis CANNOT be undone.');
}

function confirmPromote(username) {
  return confirm(
    'Promote "' +
      username +
      '" to Admin?\n\n' +
      "They will gain full access to this admin panel and all admin features.",
  );
}

function confirmDemote(username) {
  return confirm(
    'Remove admin from "' +
      username +
      '"?\n\n' +
      "They will be downgraded to a regular user.",
  );
}

// ── Search / filter ───────────────────────────────────────────────────────────

function filterTable() {
  var query = document.getElementById("searchInput").value.toLowerCase();
  var rows = document.querySelectorAll("#userTable tbody tr");

  rows.forEach(function (row) {
    var username = row.querySelector(".td-username")
      ? row.querySelector(".td-username").textContent.toLowerCase()
      : "";
    var email = row.querySelector(".td-email")
      ? row.querySelector(".td-email").textContent.toLowerCase()
      : "";

    if (username.includes(query) || email.includes(query)) {
      row.classList.remove("hidden");
    } else {
      row.classList.add("hidden");
    }
  });
}

// ── Auto-dismiss alert after 4 s ─────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function () {
  var alert = document.querySelector(".alert");
  if (alert) {
    setTimeout(function () {
      alert.style.transition = "opacity 0.5s ease";
      alert.style.opacity = "0";
      setTimeout(function () {
        alert.remove();
      }, 500);
    }, 4000);
  }
});
