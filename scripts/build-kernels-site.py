"""Refresh the published copies of the three existing standalone tools."""
from pathlib import Path
import re, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'web/kernels'
nav='''<nav aria-label="Kernel tools" style="padding:10px 20px;border-bottom:1px solid #2a2f3a;font:13px system-ui"><a href="/kernels/" style="color:#7dd3fc;text-decoration:none">← Kernel tools</a></nav>'''
sources={'qwen3-block':'assignment1/qwen3_block_visualizer.html','gemm':'visualizations/kernels/GEMM kernels visualizer.html','shared-memory':'visualizations/smem-bank-contention/visualizer.html'}
manifest={}
for slug,src in sources.items():
 p=ROOT/src;s=p.read_text()
 if 'name="viewport"' not in s:s=s.replace('<head>','<head>\n<meta name="viewport" content="width=device-width, initial-scale=1">',1)
 s=re.sub(r'(<body\b[^>]*>)',lambda m:m[0]+nav,s,count=1)
 if slug=='shared-memory':
  s=s.replace('</head>','<style>@media(max-width:760px){main{grid-template-columns:1fr}header{flex-wrap:wrap}aside{border-right:0;border-bottom:1px solid var(--line)}section{padding:12px;min-width:0}}</style></head>',1)
 target=OUT/slug/'index.html';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
 manifest[src]=hashlib.sha256(p.read_bytes()).hexdigest()
md=(ROOT/'assignment1/Problem3_block_diagram.md').read_text()
diagram=re.search(r'```mermaid\n(.*?)\n```',md,re.S).group(1)
(OUT/'qwen3-8b/diagram.mmd').write_text(diagram+'\n')
manifest['assignment1/Problem3_block_diagram.md']=hashlib.sha256(md.encode()).hexdigest()
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built three standalone tools and extracted the Qwen diagram.')
