// admin.js

function confirmDelete(username) {
  return confirm(
    'Are you sure you want to delete user "' +
      username +
      '"?\nThis cannot be undone.',
  );
}

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
