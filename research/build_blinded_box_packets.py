#!/usr/bin/env python3
"""Build self-contained, offline HTML annotation packets for the Gate 1 labels.

The exported CSV follows path_annotation_scoring.py's schema. No images or
annotations are sent to a service; the generated pages have no network calls.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import platform
import secrets
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "gate1_manual_annotation_packets_v5"
M0_ZIP = ROOT / "research" / "wan_m0_observer_labels_v1" / "blinded_annotation_package.zip"
KUBRIC = ROOT / "research" / "kubric_gate0_fixture_v1"
CSV_COLUMNS = [
    "blind_clip_id", "frame_index", "image_file", "object_visible",
    "instance_count", "identity_consistent", "bbox_x1_px", "bbox_y1_px",
    "bbox_x2_px", "bbox_y2_px", "notes",
]


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
<title>Blinded frame annotation</title>
<style>
:root{font:16px/1.45 system-ui,-apple-system,sans-serif;color:#202631;background:#f4f6f8}
*{box-sizing:border-box} body{margin:0} header{background:#172436;color:#fff;padding:14px 20px}
header h1{margin:0;font-size:19px} header p{margin:4px 0 0;color:#cbd5e1;font-size:13px}
main{max-width:1420px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
section,.panel{background:#fff;border:1px solid #dbe1e8;border-radius:10px;padding:14px;box-shadow:0 1px 2px #1522380d}
.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
button,select,input{font:inherit} button{border:1px solid #b9c4d1;background:#fff;border-radius:7px;padding:7px 11px;cursor:pointer}
button:hover{background:#eef4fa} button.primary{background:#155eef;border-color:#155eef;color:#fff} button.primary:hover{background:#0c4dd4}
button:disabled{opacity:.45;cursor:not-allowed}.status{margin-left:auto;color:#526174;font-size:13px}
.canvas-wrap{position:relative;width:max-content;max-width:100%;margin:0 auto;background:#111;line-height:0;touch-action:none}
#frameImage{display:block;max-width:min(100%,960px);max-height:64vh;width:auto;height:auto;object-fit:contain;image-rendering:pixelated}
#overlay{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none}
#boxShape{fill:#16a34a33;stroke:#20c55a;stroke-width:2;vector-effect:non-scaling-stroke}
.hint{font-size:13px;color:#526174;margin:10px 0 0}.ref{display:flex;align-items:center;gap:10px;margin:12px 0;padding:8px;background:#f6f8fa;border-radius:7px;font-size:13px;color:#435166}
#refImage{width:144px;height:84px;object-fit:contain;background:#111}.frame-strip{display:flex;gap:5px;flex-wrap:wrap;margin-top:12px}
.frame-strip button{padding:4px 7px;font-size:12px;min-width:34px}.frame-strip button.done{border-color:#16a34a;background:#eaf8ef}.frame-strip button.active{outline:2px solid #155eef}
label{display:block;font-size:13px;font-weight:600;margin:12px 0 4px;color:#334155}
select,input[type=number],input[type=text]{width:100%;border:1px solid #c7d0dc;border-radius:6px;padding:7px 8px;background:white}
.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}.note{min-height:70px;resize:vertical}
.warning{margin-top:12px;background:#fff8e7;border:1px solid #f5d88e;padding:9px;border-radius:7px;color:#5f4b12;font-size:12px}
.complete{color:#187344;font-weight:600}.muted{color:#66758a;font-size:12px}
@media(max-width:900px){main{grid-template-columns:1fr}.status{margin-left:0}#frameImage{max-height:50vh}.panel{order:-1}}
</style>
</head>
<body>
<header><h1>Blinded frame annotation</h1><p>Label each field independently for every frame. Do not infer a requested path from the motion.</p></header>
<main>
 <section>
  <div class="toolbar">
   <label for="clipSelect" style="margin:0">Sequence</label><select id="clipSelect" style="width:auto;min-width:190px"></select>
   <button id="prev">← Previous</button><button id="next">Next →</button>
   <span id="progress" class="status"></span>
  </div>
  <div id="canvasWrap" class="canvas-wrap"><img id="frameImage" alt="Current frame"><svg id="overlay" aria-label="Draw the tight visible-object bounding box"><rect id="boxShape" x="0" y="0" width="0" height="0" visibility="hidden"></rect></svg></div>
  <p class="hint">Drag around the tight visible silhouette. Use pixel-edge coordinates; exclude shadow and detached glare. Draw every frame independently; do not interpolate. Drawing a box does not set visibility, instance count, or identity: enter all four judgements separately. If absent, choose No, enter count 0, mark identity No / uncertain, and leave the box empty.</p>
  <div class="ref"><strong>Frame 0 identity reference</strong><img id="refImage" alt="Frame zero"><span id="refLabel"></span></div>
  <div id="frameStrip" class="frame-strip"></div>
 </section>
 <aside class="panel">
  <div id="currentLabel" class="muted"></div>
  <label for="visible">Target visible in this frame?</label><select id="visible"><option value="">Choose…</option><option value="true">Yes</option><option value="false">No</option></select>
  <div class="two">
   <div><label for="count">Visible instances</label><input id="count" type="number" min="0" step="1" placeholder="0, 1, …"></div>
   <div><label for="identity">Same identity as frame 0?</label><select id="identity"><option value="">Choose…</option><option value="true">Yes</option><option value="false">No / uncertain</option></select></div>
  </div>
  <label for="notes">Notes (occlusion, blur, ambiguity, duplicates)</label><textarea id="notes" class="note" placeholder="Optional notes"></textarea>
  <div class="toolbar" style="margin-top:12px"><button id="clearBox">Clear box</button><button id="save" class="primary">Download CSV</button></div>
  <div class="toolbar"><button id="import">Import/resume CSV</button><input id="importFile" type="file" accept=".csv,text/csv" hidden></div>
  <div id="validation" class="muted"></div>
  <div class="warning">This page runs locally in your browser and does not upload data. Download the CSV before closing. Keep each rater’s CSV separate until both have finished.</div>
 </aside>
</main>
<script>
const frames = __FRAME_DATA__;
const columns = __COLUMNS__;
const packetKey = __PACKET_KEY__;
const clips = [...new Set(frames.map(f => f.blind_clip_id))];
const stateKey = 'gate1-box-labels-v1-' + packetKey;
let annotations = {};
try { annotations = JSON.parse(localStorage.getItem(stateKey) || '{}'); } catch (_) { annotations = {}; }
let clipId = clips[0], frameIndex = 0, dragStart = null;
const $ = id => document.getElementById(id);
const clipFrames = id => frames.filter(f => f.blind_clip_id === id).sort((a,b)=>a.frame_index-b.frame_index);
const keyFor = f => `${f.blind_clip_id}:${f.frame_index}`;
function blank(f){return {blind_clip_id:f.blind_clip_id,frame_index:f.frame_index,image_file:f.image_file,object_visible:'',instance_count:'',identity_consistent:'',bbox_x1_px:'',bbox_y1_px:'',bbox_x2_px:'',bbox_y2_px:'',notes:''};}
for(const f of frames){const k=keyFor(f);if(!annotations[k])annotations[k]=blank(f);}
function persist(){try{localStorage.setItem(stateKey,JSON.stringify(annotations));}catch(_){} updateProgress();}
function cur(){return clipFrames(clipId).find(f=>f.frame_index===frameIndex);}
function labels(){return annotations[keyFor(cur())];}
function boxFromRow(row){let vals=['bbox_x1_px','bbox_y1_px','bbox_x2_px','bbox_y2_px'].map(k=>Number(row[k]));return vals.every(Number.isFinite)&&vals[2]>vals[0]&&vals[3]>vals[1]?vals:null;}
function drawBox(box){const r=$('boxShape');if(!box){r.setAttribute('visibility','hidden');return;}r.setAttribute('x',box[0]);r.setAttribute('y',box[1]);r.setAttribute('width',box[2]-box[0]);r.setAttribute('height',box[3]-box[1]);r.setAttribute('visibility','visible');}
function updateProgress(){const fs=clipFrames(clipId);let done=0;for(const f of fs)if(isComplete(annotations[keyFor(f)]))done++;$('progress').textContent=`${done}/${fs.length} frames complete`;$('progress').className='status '+(done===fs.length?'complete':'');}
function isComplete(r){if(!r||!['true','false'].includes(String(r.object_visible))||!['true','false'].includes(String(r.identity_consistent))||r.instance_count===''||!Number.isInteger(Number(r.instance_count))||Number(r.instance_count)<0)return false;const box=!!boxFromRow(r);if(String(r.object_visible)==='true')return Number(r.instance_count)>=1&&box;return Number(r.instance_count)===0&&String(r.identity_consistent)==='false'&&!box;}
function setField(field,value){const f=cur(),r=labels();r[field]=value;annotations[keyFor(f)]=r;persist();renderStrip();validateCurrent();}
function updateCurrent(){const f=cur(),r=labels();$('currentLabel').textContent=`${f.blind_clip_id} · frame ${String(frameIndex).padStart(2,'0')} · ${f.width}×${f.height}`;$('frameImage').src=f.data_uri;$('frameImage').width=f.width;$('frameImage').height=f.height;$('frameImage').style.width=`${Math.min(960,Math.max(640,f.width))}px`;$('frameImage').style.height='auto';$('overlay').setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);$('refImage').src=clipFrames(clipId)[0].data_uri;$('refLabel').textContent=`Frame 0 of ${clipId}`;$('visible').value=String(r.object_visible??'');$('count').value=r.instance_count??'';$('identity').value=String(r.identity_consistent??'');$('notes').value=r.notes??'';drawBox(boxFromRow(r));renderStrip();updateProgress();validateCurrent();}
function renderStrip(){const strip=$('frameStrip');strip.replaceChildren();for(const f of clipFrames(clipId)){const b=document.createElement('button');b.textContent=String(f.frame_index).padStart(2,'0');if(isComplete(annotations[keyFor(f)]))b.classList.add('done');if(f.frame_index===frameIndex)b.classList.add('active');b.title=`Frame ${f.frame_index}`;b.onclick=()=>{frameIndex=f.frame_index;updateCurrent();};strip.appendChild(b);}}
function validateCurrent(){const r=labels();let msg='';if(r.object_visible==='true'&&Number(r.instance_count)<1)msg='Visible target requires an explicit instance count ≥ 1.';else if(r.object_visible==='true'&&!boxFromRow(r))msg='Draw a box for every visible target.';else if(r.object_visible==='false'&&Number(r.instance_count)!==0)msg='For an absent target, explicitly enter instance count 0.';else if(r.object_visible==='false'&&r.identity_consistent!=='false')msg='For an absent target, explicitly mark identity No / uncertain.';else if(r.object_visible==='false'&&boxFromRow(r))msg='Clear the box for an absent target.';else if(r.object_visible===''||r.identity_consistent===''||r.instance_count==='')msg='Complete visibility, instance count, and identity independently.';else msg=isComplete(r)?'Frame label complete.':'Check the label fields.';$('validation').textContent=msg;$('validation').className=isComplete(r)?'complete':'muted';}
function move(d){const fs=clipFrames(clipId),i=fs.findIndex(f=>f.frame_index===frameIndex);frameIndex=fs[Math.max(0,Math.min(fs.length-1,i+d))].frame_index;updateCurrent();}
function csvEscape(v){const s=String(v??'');return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;}
function downloadCsv(){const lines=[columns.join(',')];for(const f of frames){const r=annotations[keyFor(f)];lines.push(columns.map(c=>csvEscape(r[c])).join(','));}const blob=new Blob(['\ufeff'+lines.join('\r\n')+'\r\n'],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='annotation_template_filled.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
function parseCsv(text){const out=[];let row=[],field='',quoted=false;for(let i=0;i<text.length;i++){const c=text[i];if(quoted){if(c==='"'&&text[i+1]==='"'){field+='"';i++;}else if(c==='"')quoted=false;else field+=c;}else if(c==='"')quoted=true;else if(c===','){row.push(field);field='';}else if(c==='\n'){row.push(field.replace(/\r$/,''));out.push(row);row=[];field='';}else field+=c;}if(field||row.length){row.push(field.replace(/\r$/,''));out.push(row);}return out;}
function importCsv(text){const rows=parseCsv(text);const head=rows.shift().map(x=>x.replace(/^\ufeff/,''));const ix=Object.fromEntries(head.map((x,i)=>[x,i]));for(const c of columns)if(ix[c]===undefined)throw Error(`Missing column: ${c}`);for(const row of rows){if(row.length===1&&!row[0])continue;const id=row[ix.blind_clip_id],idx=Number(row[ix.frame_index]),k=`${id}:${idx}`;if(!annotations[k])continue;const r=annotations[k];for(const c of columns)r[c]=row[ix[c]]??'';}persist();updateCurrent();}
for(const c of clips){const o=document.createElement('option');o.value=c;o.textContent=c;$('clipSelect').appendChild(o);}
$('clipSelect').onchange=e=>{clipId=e.target.value;frameIndex=clipFrames(clipId)[0].frame_index;updateCurrent();};
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('save').onclick=downloadCsv;$('clearBox').onclick=()=>{const r=labels();for(const k of ['bbox_x1_px','bbox_y1_px','bbox_x2_px','bbox_y2_px'])r[k]='';drawBox(null);persist();validateCurrent();};
$('visible').onchange=e=>setField('object_visible',e.target.value);$('identity').onchange=e=>setField('identity_consistent',e.target.value);$('count').oninput=e=>{const v=e.target.value;labels().instance_count=v;persist();validateCurrent();};$('notes').oninput=e=>{labels().notes=e.target.value;persist();};
$('import').onclick=()=>$('importFile').click();$('importFile').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{importCsv(await file.text());}catch(err){alert('Could not import CSV: '+err.message);}e.target.value='';};
function point(evt){const svg=$('overlay'),p=svg.createSVGPoint();p.x=evt.clientX;p.y=evt.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse());const f=cur();return [Math.max(0,Math.min(f.width,q.x)),Math.max(0,Math.min(f.height,q.y))];}
$('overlay').onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();$('overlay').setPointerCapture(e.pointerId);dragStart=point(e);drawBox([dragStart[0],dragStart[1],dragStart[0]+.1,dragStart[1]+.1]);};
$('overlay').onpointermove=e=>{if(!dragStart)return;const p=point(e),x1=Math.min(p[0],dragStart[0]),y1=Math.min(p[1],dragStart[1]),x2=Math.max(p[0],dragStart[0]),y2=Math.max(p[1],dragStart[1]);drawBox([x1,y1,x2,y2]);};
 $('overlay').onpointerup=e=>{if(!dragStart)return;const p=point(e),x1=Math.min(p[0],dragStart[0]),y1=Math.min(p[1],dragStart[1]),x2=Math.max(p[0],dragStart[0]),y2=Math.max(p[1],dragStart[1]);dragStart=null;if(x2-x1<1||y2-y1<1){drawBox(boxFromRow(labels()));return;}const r=labels();[x1,y1,x2,y2].forEach((v,i)=>r[['bbox_x1_px','bbox_y1_px','bbox_x2_px','bbox_y2_px'][i]]=Math.round(v*100)/100);drawBox([x1,y1,x2,y2]);persist();validateCurrent();renderStrip();};
window.addEventListener('keydown',e=>{if(e.target.matches('input,textarea,select'))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='s'){e.preventDefault();downloadCsv();}});
updateCurrent();
</script>
</body></html>'''


def png_frame(path: Path, clip_id: str, frame_index: int, image_file: str) -> dict[str, Any]:
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        width, height = im.size
    return {
        "blind_clip_id": clip_id,
        "frame_index": frame_index,
        "image_file": image_file,
        "width": width,
        "height": height,
        "data_uri": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
    }


def csv_text(frames: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for f in frames:
        writer.writerow({
            "blind_clip_id": f["blind_clip_id"],
            "frame_index": f["frame_index"],
            "image_file": f["image_file"],
            **{k: "" for k in CSV_COLUMNS[3:]},
        })
    return stream.getvalue()


def make_html(frames: list[dict[str, Any]], packet_key: str) -> str:
    data = json.dumps(frames, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    cols = json.dumps(CSV_COLUMNS)
    key = json.dumps(packet_key)
    return HTML.replace("__FRAME_DATA__", data).replace("__COLUMNS__", cols).replace("__PACKET_KEY__", key)


def add_packet(zip_path: Path, frames: list[dict[str, Any]], packet_key: str, readme: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("index.html", make_html(frames, packet_key))
        z.writestr("annotation_template.csv", csv_text(frames))
        z.writestr("README.txt", readme)


def read_m0_copies() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(M0_ZIP) as z:
        csv_name = next(n for n in z.namelist() if n.endswith("annotation_template.csv"))
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(csv_name), encoding="utf-8")))
        for row in rows:
            clip_id = row["blind_clip_id"]
            frame_idx = int(row["frame_index"])
            image_name = row["image_file"]
            candidates = [n for n in z.namelist() if n.endswith("/" + image_name)]
            if not candidates:
                candidates = [n for n in z.namelist() if n.endswith(f"{clip_id}/frame_{frame_idx:03d}.png")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one image for {clip_id} frame {frame_idx}, got {len(candidates)}")
            raw = z.read(candidates[0])
            with Image.open(io.BytesIO(raw)) as im:
                size = im.size
            result.setdefault(clip_id, []).append({
                "blind_clip_id": clip_id,
                "frame_index": frame_idx,
                "image_file": image_name,
                "width": size[0],
                "height": size[1],
                "data_uri": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
            })
    for items in result.values():
        items.sort(key=lambda x: x["frame_index"])
    if len(result) != 2 or any(len(x) != 17 for x in result.values()):
        raise ValueError("The M0 source packet should contain two 17-frame blinded copies")
    return result


def create_packets() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing packet directory: {OUT}")
    started = time.perf_counter()
    OUT.mkdir(parents=True)
    kubric_dir = OUT / "synthetic_reliability"
    m0_dir = OUT / "generated_sample_observer_check"
    kubric_dir.mkdir()
    m0_dir.mkdir()
    privacy_note = (
        "This packet is a local, offline annotation page. Extract the ZIP and open index.html in a modern browser. "
        "The page does not contact a server. Label every frame independently; do not interpolate boxes. "
        "Use the CSV download button before closing and return the downloaded CSV.\n\n"
        "For each frame independently record visibility, visible target-instance count, continuity with frame 0, and the "
        "tight visible-object box. Do not let one judgement fill or determine another: drawing a box never sets the three "
        "categorical fields, and changing visibility never edits count, identity, or box. Box the visible silhouette, "
        "excluding shadow and detached glare. Use continuous pixel-edge coordinates; right and bottom edges are exclusive. "
        "If absent, explicitly choose No, enter count 0, mark identity No / uncertain, and leave the box empty. Add notes "
        "for occlusion, blur, ambiguity, or duplicates. Do not look up or "
        "guess the requested path or experimental condition. Keep the CSV private until both annotators finish.\n"
    )
    keys: dict[str, Any] = {"created_for": "Gate 1 blinded annotation; keep private", "synthetic": {}, "generated_sample": {}}

    kubric_clips: list[dict[str, Any]] = []
    for direction in ("left", "right"):
        clip_id = "clip_" + secrets.token_hex(5)
        clip_dir = KUBRIC / direction / "frames"
        files = sorted(clip_dir.glob("frame_*.png"))
        if len(files) != 17:
            raise ValueError(f"Expected 17 frames in {clip_dir}, found {len(files)}")
        for path in files:
            frame_idx = int(path.stem.split("_")[-1])
            kubric_clips.append(png_frame(path, clip_id, frame_idx, f"{clip_id}/{path.name}"))
        keys["synthetic"][clip_id] = direction
    # Use the same blinded identifiers for the two independent raters.
    for rater in (1, 2):
        ordered = list(kubric_clips)
        if rater == 2:
            # Change which sequence appears first without changing the blind IDs.
            ordered = sorted(kubric_clips, key=lambda f: (keys["synthetic"][f["blind_clip_id"]] != "right", f["frame_index"]))
        add_packet(kubric_dir / f"rater_{rater:02d}.zip", ordered, f"synthetic-rater-{rater}", privacy_note)

    copies = read_m0_copies()
    source_aliases = sorted(copies)
    source_frames = copies[source_aliases[0]]
    m0_clip_id = "clip_" + secrets.token_hex(5)
    m0_frames = [
        {**f, "blind_clip_id": m0_clip_id, "image_file": f"{m0_clip_id}/frame_{f['frame_index']:03d}.png"}
        for f in source_frames
    ]
    keys["generated_sample"][m0_clip_id] = {
        "source": "one saved 17-frame Wan M0 sample",
        "source_package_aliases": source_aliases,
        "note": "Both annotators receive identical pixels under this shared blind ID; no condition or path is shown.",
    }
    for rater in (1, 2):
        add_packet(m0_dir / f"rater_{rater:02d}.zip", m0_frames, f"generated-rater-{rater}", privacy_note)

    (OUT / "PRIVATE_UNBLIND_KEY.json").write_text(json.dumps(keys, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Gate 1 offline manual annotation packets\n\n"
        "Distribute each ZIP separately to an independent rater; never send `PRIVATE_UNBLIND_KEY.json`. "
        "These v5 packets preserve v4's independent label fields and enlarge low-resolution Kubric frames for usable box annotation. "
        "Each ZIP contains one offline `index.html`, a blank CSV fallback, and instructions. The page embeds its frames, "
        "has no network calls, supports framewise tight-box drawing and CSV import/export, and does not interpolate. "
        "Images below 640 source pixels wide are enlarged to 640 CSS pixels where the viewport allows; saved coordinates "
        "remain in source-frame pixels. "
        "Visibility, instance count, identity, and box are entered independently; no action auto-fills another scored field.\n\n"
        "`synthetic_reliability/` contains the paired Kubric mirror fixture in a different clip order for the two raters. "
        "`generated_sample_observer_check/` contains separate blinded aliases of the same saved Wan M0 sample so that "
        "human boxes can be compared with the red/SAM2 proposals without showing raters those proposals.\n\n"
        "Keep returned CSVs separate and blinded until both are complete. Do not score blank templates. The Kubric labels "
        "measure annotation reliability only; the generated sample is a one-clip observer convenience check, not policy efficacy.\n",
        encoding="utf-8",
    )
    archives = sorted(OUT.rglob("*.zip"))
    manifest = {
        "packet_version": "gate1_manual_annotation_packets_v5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "build_wall_seconds": round(time.perf_counter() - started, 3),
        "build_host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0]},
        "purpose": "annotation packet generation only; no diffusion or vision inference",
        "gpu_used": False,
        "peak_vram_mb": None,
        # This is an as-built property. Any later browser smoke belongs in a
        # separate post-build QA record and must not be mistaken for rater data.
        "browser_interaction_verified": False,
        "browser_interaction_verification_scope": "at packet build time only; later QA is recorded separately",
        "source": {
            "builder": "research/build_blinded_box_packets.py",
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "annotation_independence_rule": "box drawing changes box coordinates only; categorical state labels require explicit independent entry",
            "display_scaling_rule": "display width is min(960, max(640, source_width)) CSS pixels subject to viewport max-width; SVG coordinate mapping remains in source pixels",
            "kubric_fixture": "research/kubric_gate0_fixture_v1/{left,right}/frames; one paired mirror fixture",
            "generated_sample_package_sha256": hashlib.sha256(M0_ZIP.read_bytes()).hexdigest(),
        },
        "archives": [
            {
                "path": str(p.relative_to(OUT)),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in archives
        ],
        "label_fields_populated": 0,
        "limitations": [
            "Synthetic packet contains one left/right mirror pair, not the two seed blocks required by the full reliability gate.",
            "Low-resolution images are enlarged for display only; nearest-neighbor display does not add source-image detail or change scoring coordinates.",
            "v3 is superseded because its UI coupled box drawing/visibility actions to categorical state fields; see the Gate 1 annotation-independence audit.",
            "The v4 display-size issue is corrected here; this v5 build has not yet received a browser round-trip smoke and has no human-rater reliability evidence.",
        ],
    }
    (OUT / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    create_packets()
