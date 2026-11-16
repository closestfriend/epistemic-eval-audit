"""PAPER.md -> HTML (python-markdown) -> PDF (headless Chrome), styled after the
Apart Research hackathon submission template (title between rules, centered
author block, "With Apart Research", centered abstract, sprint footnote).

Usage: uv run --no-project --with markdown python scripts/build_report_pdf.py PAPER.md PAPER.pdf
Relative image paths resolve against the PAPER.md directory.
"""
import pathlib
import re
import subprocess
import sys
import tempfile

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page {
  size: Letter;
  margin: 2.2cm 2.3cm 2.3cm 2.3cm;
  @bottom-center { content: counter(page); font: 9pt Charter, Georgia, serif; color: #555; }
}
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: Charter, "Iowan Old Style", Georgia, serif; font-size: 10.5pt;
       line-height: 1.42; color: #111; background: #fff; margin: 0; }

/* --- template title block --- */
header.titleblock { text-align: center; margin-bottom: 14pt; }
header.titleblock h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 700;
  font-size: 17pt; line-height: 1.25; letter-spacing: 0.01em; text-transform: uppercase;
  border-top: 2.5pt solid #000; border-bottom: 0.75pt solid #000;
  padding: 16pt 6pt 16pt; margin: 0 0 16pt; }
header.titleblock h1 sup { font-size: 9pt; text-transform: none; }
header.titleblock p { margin: 0 0 9pt; }
header.titleblock h2 { font-family: Georgia, serif; font-size: 12.5pt; margin: 14pt 0 6pt; }
header.titleblock p.abstract { text-align: justify; font-style: italic; max-width: 82%;
  margin: 0 auto 10pt; hyphens: auto; }
header.titleblock p.sprint-note { font-size: 8.5pt; color: #333; text-align: left;
  border-top: 0.5pt solid #999; padding-top: 4pt; margin-top: 16pt; max-width: 40%; }
header.titleblock a { color: #1a4fbf; }

/* --- body --- */
h2 { font-family: Georgia, serif; font-size: 13pt; margin: 16pt 0 5pt; break-after: avoid; }
h3 { font-family: Georgia, serif; font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
h2.newpage { break-before: page; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 7pt; padding-left: 16pt; }
li { margin-bottom: 3pt; }
hr { display: none; }
a { color: #1a4fbf; text-decoration: none; }
code { font-family: Menlo, "SF Mono", monospace; font-size: 8.3pt; background: #f2f2f2;
       padding: 0 2pt; border-radius: 2pt; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 8.5pt;
        line-height: 1.3; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 0.5pt solid #bbb; padding: 3pt 4pt; vertical-align: top; text-align: left; }
th { background: #efefef; font-weight: 600; }
td code, th code { font-size: 7.6pt; }
figure { margin: 8pt 0 10pt; break-inside: avoid; }
figure img { width: 100%; display: block; }
figcaption { font-size: 9pt; line-height: 1.35; margin-top: 4pt; color: #222; }
p.tcap { font-size: 9pt; line-height: 1.35; margin: 8pt 0 2pt; break-after: avoid; }
"""

NEWPAGE_HEADINGS = ("References", "Appendix A.")


def main(src: str, out: str) -> None:
    src_path = pathlib.Path(src).resolve()
    out_path = pathlib.Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = src_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

    # Split title block (everything before the first <hr />) from the body.
    head, sep, rest = html_body.partition("<hr />")
    if not sep:
        sys.exit("expected a --- separator after the title block")
    head = re.sub(r"<h1>(.*?)</h1>", r"<h1>\1<sup>1</sup></h1>", head, count=1, flags=re.S)
    # The abstract paragraph is the first <p> after <h2>Abstract</h2>.
    head = re.sub(r"(<h2>Abstract</h2>\s*)<p>", r'\1<p class="abstract">', head, count=1)
    head = head.replace('<p class="sprint-note">', '<p class="sprint-note"><sup>1</sup> ')
    # Table captions: paragraphs that start with <strong>Table N.</strong>
    rest = re.sub(r"<p><strong>(Table \d+\.)</strong>", r'<p class="tcap"><strong>\1</strong>', rest)
    for h in NEWPAGE_HEADINGS:
        rest = rest.replace(f"<h2>{h}", f'<h2 class="newpage">{h}', 1)

    title = re.sub(r"<.*?>", "", re.search(r"<h1>(.*?)<sup>", head, re.S).group(1)).strip()
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<base href='{src_path.parent.as_uri()}/'>"
        f"<title>{title}</title><style>{CSS}</style></head><body>"
        f"<header class='titleblock'>{head}</header><main>{rest}</main></body></html>"
    )
    html_path = out_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    work = pathlib.Path(tempfile.mkdtemp(prefix="paperpdf-chrome-"))
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--allow-file-access-from-files",
            f"--user-data-dir={work}",
            f"--print-to-pdf={out_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
