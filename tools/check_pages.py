"""Render a Markdown file to an A4 PDF with headless Chrome and print its page count.

  python3 tools/check_pages.py SOLUTION.md                 # prints e.g. "pages: 2"
  python3 tools/check_pages.py SOLUTION.md SOLUTION.pdf    # also saves the PDF

Used to keep SOLUTION.md within the assignment's two-page limit (11 pt body, 16 mm margins).
"""
import os
import re
import subprocess
import sys
import tempfile
import html
md_path = sys.argv[1]
src = open(md_path, encoding="utf-8").read().split("\n")
def inline(t):
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<i>\1</i>", t)
    return t
out, i = [], 0
while i < len(src):
    l = src[i]
    if l.startswith("|"):
        rows = []
        while i < len(src) and src[i].startswith("|"):
            if not re.match(r"^\|[-| ]+\|$", src[i]): rows.append([c.strip() for c in src[i].strip("|").split("|")])
            i += 1
        out.append("<table>" + "".join("<tr>" + "".join(f"<{'th' if j==0 else 'td'}>{inline(c)}</{'th' if j==0 else 'td'}>" for c in r) + "</tr>" for j, r in enumerate(rows)) + "</table>")
        continue
    m = re.match(r"^(#+) (.*)", l)
    if m: out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
    elif re.match(r"^(\d+\.|-) ", l):
        tag = "ol" if l[0].isdigit() else "ul"; items = []
        while i < len(src) and re.match(r"^(\d+\.|-) ", src[i]): items.append(re.sub(r"^(\d+\.|-) ", "", src[i])); i += 1
        out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>"); continue
    elif l.strip(): out.append(f"<p>{inline(l)}</p>")
    i += 1
css = "body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;line-height:1.3;margin:0} h1{font-size:16pt;margin:0 0 6px} h2{font-size:12.5pt;margin:10px 0 4px} p,li{margin:3px 0} ul,ol{margin:3px 0;padding-left:20px} table{border-collapse:collapse;margin:4px 0;font-size:10pt} td,th{border:1px solid #999;padding:2px 5px;vertical-align:top} code{font-size:9.5pt} @page{size:A4;margin:16mm}"
tmp = tempfile.mkdtemp()
html_path = os.path.join(tmp, "doc.html")
pdf_path = os.path.join(tmp, "doc.pdf")
open(html_path, "w").write(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(out)}</body></html>")

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}", "file://" + html_path], capture_output=True, timeout=120)
pages = len(re.findall(rb"/Type\s*/Page[^s]", open(pdf_path, "rb").read()))
if len(sys.argv) > 2:
    import shutil
    shutil.copy(pdf_path, sys.argv[2])
    pdf_path = sys.argv[2]
print(f"pages: {pages}  (pdf: {pdf_path})")
sys.exit(0 if pages <= 2 else 1)
