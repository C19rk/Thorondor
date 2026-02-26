// login.js

document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  const username = document.getElementById("username");
  const password = document.getElementById("password");

  // ── Password visibility toggle ────────────────────────────────────────────
  setupToggle("password");

  // ── Client-side validation ────────────────────────────────────────────────
  form.addEventListener("submit", function (e) {
    username.classList.remove("input-error");
    password.classList.remove("input-error");

    let valid = true;

    if (!username.value.trim()) {
      username.classList.add("input-error");
      username.focus();
      valid = false;
    }

    if (!password.value) {
      password.classList.add("input-error");
      if (valid) password.focus();
      valid = false;
    }

    if (!valid) e.preventDefault();
  });
});

function setupToggle(inputId) {
  const input = document.getElementById(inputId);
  const button = input && input.parentElement.querySelector(".toggle-pw");
  if (!input || !button) return;

  button.addEventListener("click", function () {
    const isHidden = input.type === "password";
    input.type = isHidden ? "text" : "password";
    button.querySelector(".icon-eye").style.display = isHidden
      ? "none"
      : "block";
    button.querySelector(".icon-eye-off").style.display = isHidden
      ? "block"
      : "none";
  });
}
