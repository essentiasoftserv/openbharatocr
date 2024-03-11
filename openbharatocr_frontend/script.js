// Define your routes and corresponding content
const routes = {
  "/home": "This is the Home page.",
  "/about": "This is the About page.",
};

// Function to render content based on the current route
function renderContent() {
  const path = window.location.pathname;
  const content = routes[path] || "404 - Page Not Found";
  document.getElementById("app").innerHTML = content;
}

// Initial content rendering
renderContent();

// Event listener for browser navigation
window.onpopstate = renderContent;
