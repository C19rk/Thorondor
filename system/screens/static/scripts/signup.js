// signup.js

document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector("form");
  const username = document.getElementById("username");
  const email = document.getElementById("email");
  const password = document.getElementById("password");
  const confirmPassword = document.getElementById("confirm_password");

  // ── Password visibility toggles ───────────────────────────────────────────
  setupToggle("password");
  setupToggle("confirm_password");

  // ── Show success popup if server flagged registration as done ─────────────
  const overlay = document.getElementById("successOverlay");
  if (overlay && overlay.dataset.show === "true") {
    overlay.classList.add("visible");
  }

  // ── Client-side validation ────────────────────────────────────────────────
  form.addEventListener("submit", function (e) {
    [username, email, password, confirmPassword].forEach((el) =>
      el.classList.remove("input-error"),
    );

    let valid = true;
    let firstBad = null;

    if (!username.value.trim()) {
      username.classList.add("input-error");
      firstBad = firstBad || username;
      valid = false;
    }

    if (!email.value.trim() || !email.value.includes("@")) {
      email.classList.add("input-error");
      firstBad = firstBad || email;
      valid = false;
    }

    if (password.value.length < 6) {
      password.classList.add("input-error");
      firstBad = firstBad || password;
      valid = false;
    }

    if (confirmPassword.value !== password.value) {
      confirmPassword.classList.add("input-error");
      firstBad = firstBad || confirmPassword;
      valid = false;
    }

    if (!valid) {
      e.preventDefault();
      if (firstBad) firstBad.focus();
    }
  });

  // ── Live confirm-password match indicator ─────────────────────────────────
  confirmPassword.addEventListener("input", function () {
    if (confirmPassword.value && confirmPassword.value !== password.value) {
      confirmPassword.classList.add("input-error");
    } else {
      confirmPassword.classList.remove("input-error");
    }
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
