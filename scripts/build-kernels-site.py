# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown==3.7"]
# ///
"""Refresh the published copies of the standalone tools.

Run with `uv run scripts/build-kernels-site.py` (needs Python-Markdown to render the SM12x report).
The SM12x attention map and report, and the TensorFold vs vLLM report, live in the
consumer-gpu-kernel-guide repository; set CONSUMER_GPU_GUIDE to its checkout if it is not at
~/Developer/projects/consumer-gpu-kernel-guide.
"""
from pathlib import Path
import re, hashlib, json, os, html
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'web/kernels'
GUIDE=Path(os.environ.get('CONSUMER_GPU_GUIDE',Path.home()/'Developer/projects/consumer-gpu-kernel-guide'))
nav='''<nav aria-label="Kernel tools" style="padding:10px 20px;border-bottom:1px solid #2a2f3a;font:13px system-ui"><a href="/kernels/" style="color:#7dd3fc;text-decoration:none">← Kernel tools</a></nav>'''
sources={'qwen3-block':'assignment1/qwen3_block_visualizer.html','gemm':'visualizations/kernels/GEMM kernels visualizer.html','shared-memory':'visualizations/smem-bank-contention/visualizer.html','flash-attention':'visualizations/flash-attention/visualizer.html'}
# Tools whose source lives in another repository: slug -> (manifest label, path)
external={'sm12x-attention':('consumer-gpu-kernel-guide:guides/sm12x-attention/visualizer.html',GUIDE/'guides/sm12x-attention/visualizer.html'),
 'tensorfold-vs-vllm':('consumer-gpu-kernel-guide:guides/tensorfold-vs-vllm/report.html',GUIDE/'guides/tensorfold-vs-vllm/report.html')}
manifest={}
def publish(slug,label,p):
 s=p.read_text()
 if 'name="viewport"' not in s:s=s.replace('<head>','<head>\n<meta name="viewport" content="width=device-width, initial-scale=1">',1)
 s=re.sub(r'(<body\b[^>]*>)',lambda m:m[0]+nav,s,count=1)
 if slug=='shared-memory':
  s=s.replace('</head>','<style>@media(max-width:760px){main{grid-template-columns:1fr}header{flex-wrap:wrap}aside{border-right:0;border-bottom:1px solid var(--line)}section{padding:12px;min-width:0}}</style></head>',1)
 if slug=='sm12x-attention':
  s=s.replace('href="REPORT.md"','href="report/"')  # the report is published as rendered HTML beside the map
 target=OUT/slug/'index.html';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
 manifest[label]=hashlib.sha256(p.read_bytes()).hexdigest()
for slug,src in sources.items():publish(slug,src,ROOT/src)
for slug,(label,p) in external.items():publish(slug,label,p)

# Render the SM12x report next to its map.
REPORT_CSS='''
:root{color-scheme:dark;--bg:#0f1115;--panel:#171a21;--ink:#e6e8ee;--dim:#8a91a3;--line:#2a2f3a;--a:#38bdf8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.top{padding:10px 20px;border-bottom:1px solid var(--line);font:13px system-ui;display:flex;gap:18px;flex-wrap:wrap}
.top a{color:#7dd3fc;text-decoration:none}
article{max-width:940px;margin:0 auto;padding:28px 24px 80px;overflow-wrap:break-word}
h1{font-size:28px;line-height:1.25;letter-spacing:-.02em;text-wrap:balance}h2{font-size:21px;margin-top:2.4em;padding-top:.6em;border-top:1px solid var(--line)}
h3{font-size:17px;margin-top:1.8em}h4{font-size:15px;margin-top:1.5em}
a{color:var(--a)}code{font:13px ui-monospace,Menlo,monospace;background:#232836;border-radius:4px;padding:1px 5px;overflow-wrap:anywhere}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto}pre code{background:none;padding:0;overflow-wrap:normal}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0 18px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{background:var(--panel);color:var(--dim);font-weight:600}tr:last-child td{border-bottom:0}
blockquote{margin:12px 0;padding:4px 16px;border-left:3px solid var(--line);color:var(--dim)}
@media(max-width:600px){article{padding:20px 16px 60px}h1{font-size:23px}table{font-size:12px}}
'''
try:
 import markdown
except ImportError:
 raise SystemExit('Python-Markdown is missing: run `uv run scripts/build-kernels-site.py`.')
rp=GUIDE/'guides/sm12x-attention/REPORT.md'
rmd=rp.read_text().replace('](visualizer.html)','](../)')
body=markdown.markdown(rmd,extensions=['tables','fenced_code','toc','sane_lists'])
body=body.replace('<table>','<div class="scroll"><table>').replace('</table>','</table></div>')
title=html.escape(re.match(r'# (.+)',rmd).group(1))
page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SM12x attention report</title><meta name="description" content="{title}"><style>{REPORT_CSS}</style></head><body><nav class="top" aria-label="Kernel tools"><a href="/kernels/">← Kernel tools</a><a href="../">SM12x support map</a></nav><article>
{body}
</article></body></html>
'''
(OUT/'sm12x-attention/report').mkdir(parents=True,exist_ok=True)
(OUT/'sm12x-attention/report/index.html').write_text(page)
manifest['consumer-gpu-kernel-guide:guides/sm12x-attention/REPORT.md']=hashlib.sha256(rp.read_bytes()).hexdigest()

md=(ROOT/'assignment1/Problem3_block_diagram.md').read_text()
diagram=re.search(r'```mermaid\n(.*?)\n```',md,re.S).group(1)
(OUT/'qwen3-8b/diagram.mmd').write_text(diagram+'\n')
manifest['assignment1/Problem3_block_diagram.md']=hashlib.sha256(md.encode()).hexdigest()
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Built {len(sources)+len(external)} standalone tools, the SM12x report, and the Qwen diagram.')
