/* preferences */
const prefs = {contrast:"normal", fontScale:1, reduced:false};

const savePrefs = () => { localStorage.setItem("ai_bridge_prefs", JSON.stringify(prefs)) }
const loadPrefs = () => {
  const raw = localStorage.getItem("ai_bridge_prefs");
  if(raw){
    const parsed = JSON.parse(raw);
    prefs.contrast = parsed.contrast ?? prefs.contrast;
    prefs.fontScale = parsed.fontScale ?? prefs.fontScale;
    prefs.reduced = parsed.reduced ?? prefs.reduced;
  }
}

/* apply UI */
const applyPrefs = () => {
  document.documentElement.style.setProperty("--font-scale", prefs.fontScale);
  if(prefs.contrast === "high"){ document.body.classList.add("high-contrast") } else { document.body.classList.remove("high-contrast") }
  if(prefs.reduced){ document.body.classList.add("reduced-motion") } else { document.body.classList.remove("reduced-motion") }
}

/* toggles */
const toggleContrast = () => {
  prefs.contrast = prefs.contrast === "high" ? "normal" : "high";
  applyPrefs();
  savePrefs();
}
const increaseFont = () => {
  prefs.fontScale = Math.min(1.4, +(prefs.fontScale + 0.1).toFixed(2));
  applyPrefs();
  savePrefs();
}
const decreaseFont = () => {
  prefs.fontScale = Math.max(0.85, +(prefs.fontScale - 0.1).toFixed(2));
  applyPrefs();
  savePrefs();
}
const toggleReducedMotion = () => {
  prefs.reduced = !prefs.reduced;
  applyPrefs();
  savePrefs();
}

/* keyboard shortcuts (do not trigger when typing) */
const handleKey = (e) => {
  const tag = document.activeElement.tagName;
  if(tag === "INPUT" || tag === "TEXTAREA" || document.activeElement.isContentEditable){ return }
  if(e.key === "c"){ toggleContrast() }
  if(e.key === "+"){ increaseFont() }
  if(e.key === "-"){ decreaseFont() }
  if(e.key === "m"){ toggleReducedMotion() }
}

/* captions demo simulation (captions.html) */
const simulateCaptions = () => {
  const out = document.getElementById("captions-output");
  if(!out) return;
  const lines = [
    "Welcome to the Accessible AI Bridge demo.",
    "This is a simulated live caption feed.",
    "Future integration: ASL→Text and Text→ASL modules.",
    "Keyboard navigation and ARIA are active on this page."
  ];
  let i = 0;
  const tick = () => {
    out.textContent = lines[i % lines.length];
    i++;
    setTimeout(tick, 2800);
  }
  tick();
}

/* init on DOM ready */
document.addEventListener("DOMContentLoaded", () => {
  loadPrefs();
  applyPrefs();
  document.addEventListener("keydown", handleKey);

  // wire buttons by id if present
  const contrastBtn = document.getElementById("toggle-contrast");
  if(contrastBtn) contrastBtn.addEventListener("click", () => toggleContrast());

  const incBtn = document.getElementById("font-increase");
  if(incBtn) incBtn.addEventListener("click", () => increaseFont());

  const decBtn = document.getElementById("font-decrease");
  if(decBtn) decBtn.addEventListener("click", () => decreaseFont());

  const motionBtn = document.getElementById("toggle-motion");
  if(motionBtn) motionBtn.addEventListener("click", () => toggleReducedMotion());

  // start captions simulation when applicable
  simulateCaptions();

  // simple form validation for contact page
  const contactForm = document.getElementById("contact-form");
  if(contactForm){
    contactForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      const name = contactForm.querySelector("#name").value.trim();
      const email = contactForm.querySelector("#email").value.trim();
      const message = contactForm.querySelector("#message").value.trim();
      const status = document.getElementById("contact-status");
      if(!name || !email || !message){
        status.textContent = "Please fill all fields.";
        status.setAttribute("role","alert");
        status.style.color = "crimson";
        return;
      }
      status.textContent = "Thank you — message received (demo).";
      status.style.color = "green";
      contactForm.reset();
    });
  }
});
