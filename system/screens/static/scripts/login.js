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

  const eyeIcon = button.querySelector(".icon-eye");
  const eyeOffIcon = button.querySelector(".icon-eye-off");

  button.addEventListener("click", function () {
    const isCurrentlyPassword = input.type === "password";

    if (isCurrentlyPassword) {
      // Switching to visible text
      input.type = "text";
      eyeIcon.style.display = "block"; // Show the "Eye" icon
      eyeOffIcon.style.display = "none"; // Hide the "Eye Off" icon
    } else {
      // Switching back to hidden password
      input.type = "password";
      eyeIcon.style.display = "none"; // Hide the "Eye" icon
      eyeOffIcon.style.display = "block"; // Show the "Eye Off" icon
    }
  });
}
