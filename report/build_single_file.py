"""Build report/mammography-report.html: index.html with styles.css and report.js inlined, for sharing.

Run after editing index.html, styles.css or report.js: uv run python build_single_file.py
"""

from pathlib import Path


HERE = Path(__file__).resolve().parent

html = (HERE / "index.html").read_text(encoding="utf-8")
css = (HERE / "styles.css").read_text(encoding="utf-8")
js = (HERE / "report.js").read_text(encoding="utf-8")
link, script = '<link rel="stylesheet" href="styles.css">', '<script src="report.js" defer></script>'
assert link in html and script in html
# the inlined script runs where report.js was deferred, so move it to the end of <body>
html = html.replace(link, f"<style>\n{css}</style>").replace(script, "")
html = html.replace("</body>", f"<script>\n{js}</script>\n</body>")
out = HERE / "mammography-report.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size / 1e3:.0f} kB")
