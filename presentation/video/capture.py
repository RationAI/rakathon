"""Screenshots of the two apps for the video: the invitation planner (predictor_ui, labels in Czech) and the
patient webapp (webapp/index.html) on a phone. Output: screens/*.png and screens/coords.js (where to tap),
both used by scene.html.

Needs the planner running: uv run python server.py (from predictor_ui/).
Run from this folder: uv run --no-project --with playwright python capture.py
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import Page, async_playwright


OUT = Path("screens")
WEBAPP = Path("../../webapp/index.html").resolve().as_uri()
PLANNER = "http://127.0.0.1:8000/"
PHONE = {"width": 390, "height": 800}  # the phone screen in the video is 390 x 844 with a 44 px status bar
BATCH = "2024-05-20"
COORDS: dict = {}

# planner labels visible in the video, English -> Czech
CZ = {
    "Rakathon 2026 · screening invitations": "Rakathon 2026 · pozvánky na screening",
    "Invitation planner": "Plánovač pozvánek",
    "Demo: a real mailing batch": "Ukázka: skutečná rozesílka",
    "Theme": "Motiv",
    "Select women": "Výběr žen",
    "Mailing batch (2024)": "Rozesílka (2024)",
    "Region": "Kraj",
    "District": "Okres",
    "Age band": "Věk",
    "Predicted chance of screening in 12 months, %": "Předpověď účasti do 12 měsíců, %",
    "PERSON_IDs (optional; one per line or comma separated)": "PERSON_ID (volitelné, jedno na řádek)",
    "Apply": "Použít",
    "All women": "Všechny ženy",
    "Reset": "Zrušit",
    "Export action list (CSV)": "Export seznamu (CSV)",
    "Predicted chance of screening in the selection": "Předpověď účasti ve výběru",
    "3 months": "3 měsíce",
    "12 months": "12 měsíců",
    "Scoring every woman with the model": "Model hodnotí každou ženu",
    "Assigning the eight segments": "Přiřazení do osmi segmentů",
    "Matching interventions and A/B tests": "Výběr zásahu a A/B testu",
    "Segments in the selection": "Segmenty ve výběru",
    "All districts": "Všechny okresy",
}
CZ_JS = """(cz) => {
  document.documentElement.lang = 'cs';
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n; (n = walk.nextNode()); ) { const t = n.nodeValue.trim(); if (cz[t]) n.nodeValue = n.nodeValue.replace(t, cz[t]); }
  document.querySelector('.lede').textContent = 'Vyberte ženy k pozvání. U každé model předpoví šanci, že na screening přijde, zařadí ji do segmentu a navrhne zásah, který na ni zabere.';
  document.querySelector('label:has(#fMin)').firstChild.nodeValue = 'od ';
  document.querySelector('label:has(#fMax)').firstChild.nodeValue = 'do ';
  document.querySelector('#fIds').placeholder = 'Vložte ID pro konkrétní seznam';
  document.querySelector('#arms').parentElement.lastChild.nodeValue = ' rozdělit do A/B skupin (50/50 v každém segmentu)';
  const day = (d) => new Date(d).toLocaleDateString('cs-CZ', { day: 'numeric', month: 'long', year: 'numeric' });
  for (const o of document.querySelectorAll('#fBatch option'))
    o.textContent = o.value ? `${day(o.value)} · ${(+o.textContent.split('·')[1].replace(/\\D/g, '')).toLocaleString('cs-CZ')} dopisů` : 'Libovolná rozesílka';
}"""
# re-rendered by the page after every selection, so translated after each one
LIVE_JS = """() => {
  const labels = ['vybraných žen', 'průměrná šance do 12 měsíců', 'čekaných vyšetření bez zásahu', 'nejspíš nepřijde'];
  const cz = (s) => s.replace(/(\\d),(\\d)/g, '$1\\u202f$2').replace(/(\\d)\\.(\\d)/g, '$1,$2');
  document.querySelectorAll('#kpis .kpi').forEach((k, i) => { k.querySelector('span').textContent = labels[i];
    const b = k.querySelector('b'); b.textContent = cz(b.textContent); });
  document.querySelectorAll('#hist text').forEach((t) => { t.textContent = cz(t.textContent.replace('mean', 'průměr')); });
  const n = document.querySelector('#histNote b');
  if (n) document.querySelector('#histNote').innerHTML = `Ženy podle předpovězené šance, že přijdou do 12 měsíců. <b>${cz(n.textContent)}</b> z nich má šanci pod 10 %: na běžný dopis nejspíš nezareagují.`;
}"""


async def box(page: Page, selector: str, nth: int = 0) -> dict:
    b = await page.locator(selector).nth(nth).bounding_box()
    return {k: round(v, 1) for k, v in b.items()}


async def planner(browser) -> None:
    page = await browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="light")
    await page.goto(PLANNER)
    await page.wait_for_selector("#kpis .kpi")
    await page.evaluate("document.fonts.ready")
    await page.evaluate(CZ_JS, CZ)
    # the selection: one real mailing batch, women under 10 % predicted chance of coming
    await page.select_option("#fBatch", BATCH)
    await page.fill("#fMax", "10")
    await page.wait_for_timeout(1500)
    await page.evaluate(LIVE_JS)
    await page.screenshot(path=OUT / "planner_before.png")
    COORDS["apply"] = await box(page, "#apply")
    # after Apply: the three pipeline stages done and the plan-ready toast (no scrolling, so the frame stays put)
    await page.evaluate("() => { Element.prototype.scrollIntoView = () => {}; }")
    await page.click("#apply")
    await page.wait_for_timeout(2000)
    await page.evaluate(LIVE_JS)
    await page.evaluate("""() => { const t = document.querySelector('#toast'), b = t.querySelectorAll('b');
      t.innerHTML = `✓ Plán připraven: <b>${b[0].textContent.replace(/,/g, '\\u202f')}</b> žen · <b>${b[1].textContent}</b> segmentů · zprávy jsou připravené k odeslání`;
      clearTimeout(window.toast.h); t.classList.add('on'); }""")
    await page.wait_for_timeout(500)
    await page.screenshot(path=OUT / "planner_after.png")
    COORDS["toast"] = await box(page, "#toast")
    COORDS["women"] = await page.evaluate("document.querySelector('#kpis .kpi b').textContent")
    await page.close()


async def settle(page: Page, ms: int = 1600) -> None:
    await page.evaluate("document.fonts.ready")
    await page.wait_for_timeout(ms)


async def webapp(browser) -> None:
    page = await browser.new_page(viewport=PHONE, device_scale_factor=3, color_scheme="light")
    await page.goto(WEBAPP)
    await settle(page)
    # the whole home page, scrolled through once so the survival bars have animated in
    for y in range(0, 4000, 300):
        await page.evaluate(f"window.scrollTo(0, {y})")
        await page.wait_for_timeout(120)
    await page.evaluate("window.scrollTo(0, 0)")
    await settle(page, 2000)
    await page.evaluate("document.querySelector('#nav').style.position = 'relative'")
    height = await page.evaluate("document.documentElement.scrollHeight")
    await page.screenshot(path=OUT / "web_home_full.png", full_page=True, clip={"x": 0, "y": 0, "width": PHONE["width"], "height": height})
    await page.evaluate("document.querySelector('#nav').style.position = ''")
    COORDS["home_height"] = height
    COORDS["tile_fear"] = await page.evaluate("""(() => { const r = [...document.querySelectorAll('#tiles .tile')][2].getBoundingClientRect();
      return { x: r.x, y: r.y + scrollY, width: r.width, height: r.height }; })()""")
    # fear flow, step 3: "Zavolání zpět není diagnóza"
    await page.evaluate("flow('strach', 2)")
    await settle(page, 2600)
    await page.screenshot(path=OUT / "web_check.png")
    COORDS["navbook"] = await box(page, "#navbook")
    # booking: earliest centre in Jihomoravský kraj, then the third free slot
    await page.evaluate("""() => { S.region = 'Jihomoravský kraj';
      S.center = CENTERS.filter((c) => c.region === S.region).sort((a, b) => (a.wait ?? 999) - (b.wait ?? 999))[0]; S.date = null; S.time = null; schedTime(); }""")
    await settle(page)
    await page.screenshot(path=OUT / "web_slot0.png")
    COORDS["slot"] = await box(page, ".slot:not([disabled])", 2)
    await page.locator(".slot:not([disabled])").nth(2).click()
    await page.wait_for_timeout(500)
    await page.screenshot(path=OUT / "web_slot.png")
    COORDS["continue"] = await box(page, ".sumbar .btn")
    await page.evaluate("() => { S.name = 'Jana Nováková'; S.phone = '+420 777 123 456'; schedDone(); }")
    await settle(page, 2200)
    await page.screenshot(path=OUT / "web_done.png")
    await page.close()


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome")
        await webapp(browser)
        await planner(browser)
        await browser.close()
    (OUT / "coords.js").write_text(f"window.COORDS = {json.dumps(COORDS, ensure_ascii=False, indent=1)};\n", encoding="utf-8")
    print(json.dumps(COORDS, ensure_ascii=False))


asyncio.run(main())
