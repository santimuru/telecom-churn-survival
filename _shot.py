"""Visual QA: click each top-nav section, scroll-tile screenshots to catch render bugs."""
import sys, time, os
from playwright.sync_api import sync_playwright

OUT = "C:/Users/biconsulting/portfolio/telecom-churn-survival/_qa"
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

URL = "http://localhost:8601/"
NAV = ["Brief", "Survival", "Model & Drivers", "Simulator", "Impact"]
VH = 1000

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": VH})
    pg.goto(URL, wait_until="networkidle", timeout=60000)
    time.sleep(4)
    for si, label in enumerate(NAV):
        clicked = pg.evaluate("""(lbl) => {
          const labels=[...document.querySelectorAll('div[data-testid="stRadio"] label')];
          const el=labels.find(e=>e.innerText.trim().toLowerCase()===lbl.toLowerCase());
          if(el){el.click(); return true;} return false; }""", label)
        time.sleep(3.2)
        cont = pg.query_selector('[data-testid="stMainBlockContainer"]')
        total = pg.evaluate("(e)=>e.scrollHeight", cont) if cont else VH
        n = max(1, (total // VH) + 1)
        slug = label.replace(" ", "_").replace("&", "and")
        for i in range(n):
            pg.evaluate(f"()=>document.querySelector('[data-testid=stMain]').scrollTo(0,{i*VH})")
            time.sleep(0.9)
            pg.screenshot(path=f"{OUT}/{si:02d}_{slug}_{i:02d}.png")
        print(f"{label}: clicked={clicked} {n} tiles ({total}px)")
    b.close()
