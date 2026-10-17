// Shared by the video scenes (pozvanka.html, vysetreni.html, preziti.html): timing helpers, the rise-in effect
// of the patient webapp, and the flat SVG women. Each scene builds its DOM, defines render(t) and calls boot().
// Nothing moves on its own (no CSS animations): every frame is a pure function of t, so render.py can take any frame.

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const E = { lin: (x) => x, out: (x) => 1 - (1 - x) ** 3, in: (x) => x ** 3, io: (x) => (x < 0.5 ? 4 * x ** 3 : 1 - (-2 * x + 2) ** 3 / 2),
  back: (x) => 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2 };
const P = (t, a, b, e = E.io) => e(clamp((t - a) / (b - a)));          // progress 0..1 between a and b
const L = (a, b, p) => a + (b - a) * p;
const K = (t, keys) => {                                               // keyframes [[time, value, easing to next], ...]
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) if (t <= keys[i][0]) {
    const [t0, v0, e = "io"] = keys[i - 1], [t1, v1] = keys[i], p = E[e]((t - t0) / (t1 - t0));
    return Array.isArray(v0) ? v0.map((x, j) => L(x, v1[j], p)) : L(v0, v1, p);
  }
  return keys[keys.length - 1][1];
};
const win = (t, a, b, d = 0.35) => Math.min(P(t, a, a + d, E.out), 1 - P(t, b - d, b, E.in));  // fade in at a, out by b
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const css = (el, o) => { for (const k in o) el.style[k] = o[k]; };
const nb = (s) => s.replace(/(^|\s)([aikosuvzAIKOSUVZ]) /g, "$1$2 ");  // Czech: no one-letter word at a line end

// the webapp's "rise": up 24 px out of a blur; out by b, a little upwards
function rise(el, t, a, b, d = 0) {
  const p = P(t, a + d, a + d + 0.7, E.out), q = P(t, b - 0.3, b, E.in);
  css(el, { opacity: String(p * (1 - q)), transform: `translateY(${24 * (1 - p) - 16 * q}px)`, filter: p < 1 ? `blur(${6 * (1 - p)}px)` : "none" });
}
// scenes [{id, a, b}]: shown between a and b; their children with data-d rise in with that delay
function scenes(t, list) {
  for (const s of list) {
    const el = $("#" + s.id), on = t >= s.a && t < s.b;
    el.style.display = on ? "" : "none";
    if (on) $$("[data-d]", el).forEach((c) => rise(c, t, s.a, s.b, +c.dataset.d));
  }
}

// ---------------------------------------------------------------- the women (flat SVG busts, viewBox 400 x 560)
function woman(o, cls = "woman") {
  return `<svg class="${cls}" viewBox="0 0 400 560">
  <path d="${o.hairBack}" fill="${o.hair}"/>
  <path d="M170 262 L230 262 L236 354 C220 372 180 372 164 354 Z" fill="${o.shade}"/>
  <path d="M40 560 C44 430 92 368 166 348 C180 372 220 372 234 348 C308 368 356 430 360 560 Z" fill="${o.top}"/>
  <path d="M166 348 C176 392 224 392 234 348 C226 360 174 360 166 348 Z" fill="${o.shade}"/>
  ${o.extra || ""}
  <ellipse cx="127" cy="212" rx="12" ry="18" fill="${o.shade}"/><ellipse cx="273" cy="212" rx="12" ry="18" fill="${o.shade}"/>
  <ellipse cx="200" cy="200" rx="74" ry="88" fill="${o.skin}"/>
  <circle cx="165" cy="236" r="13" fill="#F07A8C" opacity=".22"/><circle cx="235" cy="236" r="13" fill="#F07A8C" opacity=".22"/>
  <g class="eyes"><ellipse cx="172" cy="208" rx="6.5" ry="7.5" fill="#2A1D1F"/><ellipse cx="228" cy="208" rx="6.5" ry="7.5" fill="#2A1D1F"/></g>
  <path d="M157 189 Q172 181 187 188 M213 188 Q228 181 243 189" stroke="${o.brow}" stroke-width="5" stroke-linecap="round" fill="none"/>
  <path d="M200 214 Q194 234 204 238" stroke="${o.shade}" stroke-width="3.5" stroke-linecap="round" fill="none"/>
  <path class="m0" d="M184 258 Q200 262 216 258" stroke="#A8455A" stroke-width="5" stroke-linecap="round" fill="none"/>
  <path class="m1" d="M178 253 Q200 274 222 253" stroke="#A8455A" stroke-width="5" stroke-linecap="round" fill="none" opacity="0"/>
  <path d="${o.hairFront}" fill="${o.hair}"/>
  ${o.glasses ? `<g stroke="#5B3B5E" stroke-width="5" fill="rgba(255,255,255,.18)"><rect x="148" y="192" width="48" height="36" rx="13"/><rect x="204" y="192" width="48" height="36" rx="13"/><path d="M196 206 Q200 202 204 206" fill="none"/></g>` : ""}
  ${o.earring ? `<circle cx="126" cy="238" r="6" fill="#E8B84A"/><circle cx="274" cy="238" r="6" fill="#E8B84A"/>` : ""}
</svg>`;
}
const JANA = { skin: "#F3C7AB", shade: "#E2A88B", hair: "#4A2C2A", brow: "#3A2220", top: "#E2557A", earring: true,
  hairBack: "M110 212 C98 112 158 92 204 94 C266 97 304 140 292 222 C286 268 304 322 290 356 L112 356 C96 318 116 266 110 212 Z",
  hairFront: "M124 196 C118 124 168 100 210 103 C258 106 288 142 280 194 C262 154 226 134 192 134 C176 160 150 178 124 196 Z",
  extra: `<path d="M150 352 L200 420 L250 352" fill="none" stroke="#C9405F" stroke-width="6" stroke-linejoin="round"/>` };
const MARIE = { skin: "#EEC2A4", shade: "#D9A386", hair: "#C9C4D6", brow: "#8E879E", top: "#2E8C86", glasses: true,
  hairBack: "M118 206 C110 120 160 96 204 98 C258 100 296 132 286 210 C282 236 278 252 270 262 L130 262 C122 248 118 230 118 206 Z",
  hairFront: "M122 200 C114 128 166 102 208 104 C256 106 290 138 280 192 C270 168 248 150 222 144 C196 150 166 150 140 168 C130 178 124 188 122 200 Z",
  extra: `<path d="M166 348 L200 470 L234 348 C226 360 174 360 166 348 Z" fill="#EAF4F3"/><path d="M136 356 L182 520 M264 356 L218 520" stroke="#24756F" stroke-width="6" stroke-linecap="round"/>` };
const PETRA = { skin: "#E9BC9C", shade: "#D49C7C", hair: "#9A4A2A", brow: "#7A3820", top: "#7C6CF0", earring: true,
  hairBack: "M200 46 a44 44 0 1 0 0.1 0 Z M118 214 C108 128 158 100 200 100 C246 100 292 128 282 214 C278 238 272 252 266 262 L134 262 C128 252 122 238 118 214 Z",
  hairFront: "M124 198 C118 130 162 104 202 104 C246 104 286 132 278 196 C266 160 240 140 214 136 C204 150 168 162 124 198 Z",
  extra: `<path d="M166 348 C180 380 220 380 234 348" fill="none" stroke="#5E4FD0" stroke-width="7" stroke-linecap="round"/>` };

function blink(svg, t, off) {
  const k = (t + off) % 3.4, b = k < 0.14 ? 0.12 : 1;
  css($(".eyes", svg), { transform: `scaleY(${b})`, transformOrigin: "200px 208px" });
}

// render.py reads render, DURATION, SIZE, CUES (sound) and MUSIC (optional) from the page
function boot(render, duration, size = [1920, 1080]) {
  window.render = render;
  window.DURATION = duration;
  window.SIZE = size;
  window.ready = Promise.all([document.fonts.ready, ...[...document.images].map((i) => i.decode().catch(() => {}))]);
  render(+(new URLSearchParams(location.search).get("t") || 0));
}
