"""Single-file HTML browser. No server, no network: the atlas JSON is embedded in the page."""
from __future__ import annotations

import json

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>neuron atlas: __TITLE__</title>
<style>
body{font:15px/1.45 system-ui,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}
header{padding:14px 20px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0}
h1{font-size:18px;margin:0 0 4px}.meta{color:#555;font-size:13px}
.bar{margin-top:8px;display:flex;gap:10px;flex-wrap:wrap}input,select{font:inherit;padding:4px 8px}
main{padding:14px 20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 12px}
.card h2{font-size:14px;margin:0 0 4px}.tt{color:#444;font-size:13px;margin-bottom:6px}
.ex{font:13px/1.4 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;margin:2px 0}
.ex b{background:#ffe066;padding:0 1px}.a{color:#888}
</style></head><body>
<header><h1 id="h"></h1><div class="meta" id="m"></div>
<div class="bar"><input id="q" placeholder="filter by token (e.g. the)" aria-label="filter by token">
<select id="s" aria-label="sort"><option value="max">sort: highest max activation</option><option value="id">sort: neuron id</option><option value="mean">sort: highest mean activation</option></select>
<span class="meta" id="c"></span></div></header>
<main id="main"></main>
<script type="application/json" id="atlas">__DATA__</script>
<script>
var A=JSON.parse(document.getElementById('atlas').textContent);
function vis(s){return s.replace(/\\n/g,'\\u21b5');}
document.getElementById('h').textContent='Neuron atlas: '+A.model+', layer '+A.layer;
document.getElementById('m').textContent=A.neurons_included+' of '+A.n_neurons+' neurons, '+A.n_texts+' texts, '+A.n_tokens+' tokens ranked, top '+A.top_k+' contexts each. Activations are post-nonlinearity MLP values.';
function render(){
  var q=document.getElementById('q').value.trim().toLowerCase(),s=document.getElementById('s').value,list=A.neurons.slice();
  if(q)list=list.filter(function(n){return n.examples.some(function(e){return e.token.toLowerCase().indexOf(q)>=0;});});
  list.sort(function(a,b){return s==='id'?a.id-b.id:s==='mean'?b.mean_act-a.mean_act||a.id-b.id:b.max_act-a.max_act||a.id-b.id;});
  var main=document.getElementById('main');main.textContent='';
  document.getElementById('c').textContent=list.length+' match'+(list.length>300?' (first 300 drawn, narrow the filter)':'');
  list.slice(0,300).forEach(function(n){
    var c=document.createElement('div');c.className='card';
    var h=document.createElement('h2');h.textContent='neuron '+n.id+'  (max '+n.max_act+', mean '+n.mean_act+')';c.appendChild(h);
    var t=document.createElement('div');t.className='tt';
    t.textContent='top tokens: '+n.top_tokens.map(function(x){return JSON.stringify(x.token)+' x'+x.count;}).join(', ');c.appendChild(t);
    n.examples.forEach(function(e){
      var d=document.createElement('div');d.className='ex';
      var a=document.createElement('span');a.className='a';a.textContent=e.act.toFixed(2)+'  ';d.appendChild(a);
      d.appendChild(document.createTextNode(vis(e.before)));
      var b=document.createElement('b');b.textContent=vis(e.token);d.appendChild(b);
      d.appendChild(document.createTextNode(vis(e.after)));c.appendChild(d);});
    main.appendChild(c);});
}
document.getElementById('q').addEventListener('input',render);document.getElementById('s').addEventListener('change',render);render();
</script></body></html>
"""


def render_html(atlas: dict) -> str:
    data = json.dumps(atlas, ensure_ascii=False, separators=(",", ":"))
    # keep the JSON from closing the script element or starting an HTML comment
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    title = str(atlas.get("model", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
