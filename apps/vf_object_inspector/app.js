import {EDGES,FOOTPRINT,cameraSegments,projectedBounds,bestCamera,orbitBasis,orbitProject,bevProject} from './geometry.mjs';
const $=id=>document.getElementById(id);
const COLORS={gt:'#65dca0',baseline:'#ffc16a',detzero:'#7ab9ff'};
const NAMES={gt:'Ground truth',baseline:'BEVFusion',detzero:'BEVFusion + DetZero'};
const state={meta:null,data:null,frame:30,selected:null,instance:null,focus:true,points:new Float32Array(),image:null,imageKey:null,cameraName:null,version:0,abort:null,
 cam:{zoom:1,pan:[0,0],fit:null},bev:{center:[0,0,0],radius:50},three:{target:[0,0,0],distance:110,yaw:-2.1,pitch:.65}};
const imageCache=new Map();let playTimer=null,drawQueued=false;
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=3)=>v==null?'—':Number(v).toFixed(n);
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('hidden',!text);$('message').classList.toggle('error',error);}
function selected(){return state.data?.objects.find(o=>o.id===state.selected);}
function selectedRecords(){const o=selected();if(!o)return {};return {gt:state.data.gt.find(r=>r.id===o.gt),baseline:state.data.baseline.find(r=>r.source_index===o.source_index),detzero:state.data.detzero.find(r=>r.source_index===o.source_index)};}
function classFits(r){const c=$('classFilter').value;return c==='all'||(c==='car'?r.eval_class==='car':r.class===c);}
function rows(){if(!state.data)return [];const query=$('search').value.toLowerCase(),filter=$('objectFilter').value;
 return state.data.objects.filter(o=>classFits(o)&&(!query||`${o.class} ${o.id} ${o.source_index??''} ${o.instance??''}`.toLowerCase().includes(query))&&
   (filter==='all'||filter==='gain'&&o.delta_iou>1e-5||filter==='loss'&&o.delta_iou< -1e-5||filter==='unpaired'&&(o.gt==null||o.source_index==null)));}
function visibleGroups(){if(!state.data)return {gt:[],baseline:[],detzero:[]};const selectedBoxes=selectedRecords();const result={};
 for(const method of Object.keys(COLORS)){
  const checked=$({gt:'showGT',baseline:'showBaseline',detzero:'showDetzero'}[method]).checked;
  result[method]=!checked?[]:state.focus?(selectedBoxes[method]?[selectedBoxes[method]]:[]):state.data[method].filter(classFits);
 }return result;}
function requestDraw(){if(drawQueued)return;drawQueued=true;requestAnimationFrame(()=>{drawQueued=false;drawCamera();drawBev();drawThree();});}
function canvasContext(id){const canvas=$(id),rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);const w=Math.max(1,rect.width),h=Math.max(1,rect.height);
 if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);return {ctx,w,h,canvas};}
function styleLine(ctx,method,focused=false){ctx.strokeStyle=COLORS[method];ctx.lineWidth=method==='gt'?(focused?3.5:2):method==='baseline'?(focused?2.5:1.5):(focused?2:1.3);ctx.setLineDash(method==='baseline'?[7,5]:[]);ctx.globalAlpha=focused||state.focus?1:.65;}
function chosenCamera(){if(!state.data)return null;const choice=$('camera').value;return choice==='AUTO'?(bestCamera(selectedRecords().gt||selectedRecords().baseline||state.data.gt[0],state.data.cameras)||Object.keys(state.data.cameras)[0]):choice;}
async function loadCamera(){if(!state.data)return;const cameraName=chosenCamera(),cam=state.data.cameras[cameraName];if(!cam)return;
 state.cameraName=cameraName;$('cameraTitle').textContent=cameraName;$('cameraLoading').hidden=false;$('cameraLoading').textContent='Đang tải ảnh camera…';
 const key=cam.image_url;state.imageKey=key;state.image=null;requestDraw();
 try{let image=imageCache.get(key);if(!image){image=await new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('Không đọc được ảnh camera. Kiểm tra sensor-root.'));img.src=key;});imageCache.set(key,image);if(imageCache.size>24)imageCache.delete(imageCache.keys().next().value);}
  if(state.imageKey!==key)return;state.image=image;$('cameraLoading').hidden=true;resetCamera();requestDraw();
 }catch(error){if(state.imageKey===key){$('cameraLoading').textContent=error.message;message(error.message,true);}}}
function resetCamera(){state.cam.zoom=1;state.cam.pan=[0,0];state.cam.fit=null;requestDraw();}
function resetSpatial(){const record=selectedRecords().gt||selectedRecords().baseline;
 const center=state.focus&&record?record.center:[0,0,0];const radius=state.focus&&record?Math.max(3.3,Math.max(...record.box.size)*1.15+1):50;
 state.bev={center:[...center],radius};state.three={target:[...center],distance:state.focus?radius*2.5:110,yaw:-2.1,pitch:.65};requestDraw();}
function cameraFit(w,h,cam){let box=null;if(state.focus){const records=Object.values(selectedRecords()).filter(Boolean);const bounds=records.map(r=>projectedBounds(r,cam)).filter(Boolean);
 if(bounds.length){const low=[Math.min(...bounds.map(b=>b[0])),Math.min(...bounds.map(b=>b[1]))],high=[Math.max(...bounds.map(b=>b[2])),Math.max(...bounds.map(b=>b[3]))];
  const bw=Math.max(180,high[0]-low[0]),bh=Math.max(140,high[1]-low[1]),cx=(low[0]+high[0])/2,cy=(low[1]+high[1])/2;box=[cx-bw*.78,cy-bh*.78,bw*1.56,bh*1.56];}}
 if(!box)box=[0,0,cam.width,cam.height];const scale=Math.min(w/box[2],h/box[3])*state.cam.zoom;
 return {scale,x:w/2-(box[0]+box[2]/2)*scale+state.cam.pan[0],y:h/2-(box[1]+box[3]/2)*scale+state.cam.pan[1]};}
function drawCamera(){const {ctx,w,h}=canvasContext('cameraCanvas');ctx.fillStyle='#090e19';ctx.fillRect(0,0,w,h);if(!state.data||!state.image)return;
 const cam=state.data.cameras[state.cameraName],fit=cameraFit(w,h,cam);state.cam.fit=fit;ctx.drawImage(state.image,fit.x,fit.y,cam.width*fit.scale,cam.height*fit.scale);
 const chosen=selectedRecords();for(const [method,records] of Object.entries(visibleGroups()))for(const r of records){
  const focused=r.id===chosen[method]?.id;styleLine(ctx,method,focused);
  for(const [a,b] of cameraSegments(r,cam)){ctx.beginPath();ctx.moveTo(fit.x+a[0]*fit.scale,fit.y+a[1]*fit.scale);ctx.lineTo(fit.x+b[0]*fit.scale,fit.y+b[1]*fit.scale);ctx.stroke();}
 }
 ctx.globalAlpha=1;ctx.setLineDash([]);
 const target=chosen.gt||chosen.baseline;if(target&&!projectedBounds(target,cam)){
  ctx.fillStyle='#0b1120dc';ctx.fillRect(16,h-46,w-32,30);ctx.fillStyle='#dce8f8';ctx.font='12px system-ui';ctx.fillText('Object không nằm trong camera này. Chọn camera khác hoặc chế độ tự chọn.',26,h-26);}
}
function pointColor(z){return z<-.3?'#566477':z<.8?'#647c8d':z<2?'#8c99a5':'#9ba6af';}
function drawBev(){const {ctx,w,h}=canvasContext('bevCanvas');ctx.fillStyle='#090f1b';ctx.fillRect(0,0,w,h);const {center,radius}=state.bev;const pixel=p=>bevProject(p,center,radius,w,h);
 const step=radius<8?1:10;ctx.lineWidth=1;ctx.strokeStyle='#233249';ctx.font='9px ui-monospace';ctx.fillStyle='#91a4bd';
 for(let axis=0;axis<2;axis++)for(let v=Math.ceil((center[axis]-radius)/step)*step;v<=center[axis]+radius;v+=step){
  const a=[...center],b=[...center];a[axis]=v;b[axis]=v;a[1-axis]-=radius*1.5;b[1-axis]+=radius*1.5;const pa=pixel(a),pb=pixel(b);ctx.beginPath();ctx.moveTo(...pa);ctx.lineTo(...pb);ctx.stroke();
  if(axis===0){const q=pixel([v,center[1],0]);ctx.fillText(`${v.toFixed(0)}m`,5,q[1]-3);}}
 if($('showPoints').checked){for(let i=0;i<state.points.length;i+=3){const p=state.points.subarray(i,i+3),q=pixel(p);if(q[0]<0||q[0]>w||q[1]<0||q[1]>h)continue;ctx.fillStyle=pointColor(p[2]);ctx.fillRect(q[0],q[1],1.2,1.2);}}
 const chosen=selectedRecords();for(const [method,records] of Object.entries(visibleGroups()))for(const r of records){styleLine(ctx,method,r.id===chosen[method]?.id);ctx.beginPath();FOOTPRINT.forEach((idx,n)=>{const q=pixel(r.corners[idx]);n?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();ctx.stroke();}
 ctx.globalAlpha=1;ctx.setLineDash([]);const ego=pixel([0,0,0]);ctx.fillStyle='#e9f1fb';ctx.beginPath();ctx.moveTo(ego[0],ego[1]-7);ctx.lineTo(ego[0]-5,ego[1]+5);ctx.lineTo(ego[0]+5,ego[1]+5);ctx.fill();
 ctx.font='10px ui-monospace';ctx.fillStyle='#dce8fa';ctx.fillText(`±${radius.toFixed(1)} m`,w-72,17);
}
function drawThree(){const {ctx,w,h}=canvasContext('threeCanvas');ctx.fillStyle='#090f1b';ctx.fillRect(0,0,w,h);const view=state.three,basis=orbitBasis(view.target,view.distance,view.yaw,view.pitch),project=p=>orbitProject(p,basis,w,h);
 const radius=state.focus?state.bev.radius*1.5:60,step=state.focus?1:10;
 const ground=selectedRecords().gt?.center[2]-(selectedRecords().gt?.box.size[2]||0)/2;
 // Grid is a horizontal reference through the GT bottom (or ego Z=0), not a fitted road plane.
 const z=state.focus&&Number.isFinite(ground)?ground:0;ctx.strokeStyle='#233249';ctx.lineWidth=1;
 for(let axis=0;axis<2;axis++)for(let v=Math.ceil((view.target[axis]-radius)/step)*step;v<=view.target[axis]+radius;v+=step){const a=[...view.target],b=[...view.target];a[axis]=b[axis]=v;a[1-axis]-=radius;b[1-axis]+=radius;a[2]=b[2]=z;const pa=project(a),pb=project(b);if(pa&&pb){ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();}}
 if($('showPoints').checked){for(let i=0;i<state.points.length;i+=3){const p=state.points.subarray(i,i+3);if(state.focus&&Math.hypot(p[0]-view.target[0],p[1]-view.target[1])>radius)continue;const q=project(p);if(!q||q[0]<0||q[0]>w||q[1]<0||q[1]>h)continue;ctx.fillStyle=pointColor(p[2]);const size=Math.max(.8,Math.min(2.4,18/q[2]));ctx.fillRect(q[0],q[1],size,size);}}
 const chosen=selectedRecords();for(const [method,records] of Object.entries(visibleGroups()))for(const r of records){styleLine(ctx,method,r.id===chosen[method]?.id);const pts=r.corners.map(project);for(const [a,b] of EDGES){if(!pts[a]||!pts[b])continue;ctx.beginPath();ctx.moveTo(pts[a][0],pts[a][1]);ctx.lineTo(pts[b][0],pts[b][1]);ctx.stroke();}}
 ctx.globalAlpha=1;ctx.setLineDash([]);const start=project(view.target);if(start){for(const [i,c,label] of [[0,'#ff9898','X'],[1,'#65dca0','Y'],[2,'#7ab9ff','Z']]){const p=[...view.target];p[i]+=state.focus?1:5;const q=project(p);if(!q)continue;ctx.strokeStyle=c;ctx.beginPath();ctx.moveTo(start[0],start[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.fillStyle=c;ctx.font='10px ui-monospace';ctx.fillText(label,q[0]+3,q[1]);}}
 ctx.fillStyle='#91a4bd';ctx.font='9px system-ui';ctx.fillText(state.focus?'Grid: đáy GT tham chiếu, không phải road plane':'Grid: ego Z = 0',10,16);
}
function renderList(){const list=rows();$('objectCount').textContent=`${list.length} objects`;const t=state.data?.iou_threshold||.5;
 $('objects').innerHTML=list.map(o=>`<tr data-id="${esc(o.id)}" class="${o.id===state.selected?'selected':''}"><td><button><span class="object-name">${esc(o.class)}</span><span class="mini-id">${o.kind==='gt'?'GT '+esc(o.id.slice(2,10)):'Prediction #'+o.source_index}</span></button></td><td>${o.source_index==null?'—':'#'+o.source_index}</td><td class="${o.iou_baseline==null?'muted':o.iou_baseline>=t?'good':'bad'}">${fmt(o.iou_baseline)}</td><td class="${o.iou_detzero==null?'muted':o.iou_detzero>=t?'good':'bad'}">${fmt(o.iou_detzero)}</td><td class="${o.delta_iou==null?'muted':o.delta_iou>=0?'good':'bad'}">${o.delta_iou==null?'—':(o.delta_iou>=0?'+':'')+fmt(o.delta_iou)}</td></tr>`).join('')||'<tr><td colspan="5" class="empty">Không có object theo bộ lọc.</td></tr>';
 if(state.data){const count=method=>state.data[method].filter(r=>classFits(r)&&state.data.matches[method==='detzero'?'detzero':'baseline'][r.source_index]!=null).length;
 $('tpCount').textContent=`TP trong bộ lọc: ${count('baseline')} → ${count('detzero')}`;}
}
function renderDetail(){const o=selected(),records=selectedRecords();if(!o){$('detailTitle').textContent='Chưa chọn object';$('detailBody').innerHTML='<p class="muted">Chọn một hàng hoặc bấm box trên ảnh/BEV.</p>';$('selectedBadge').textContent='Chọn object bên dưới';return;}
 $('detailTitle').textContent=o.class;$('objectId').textContent=o.source_index==null?'GT only':`SOURCE #${o.source_index}`;$('selectedBadge').textContent=`${o.class.toUpperCase()} · ${o.gt?'GT '+o.gt.slice(2,10):'prediction'} · ${o.source_index==null?'no pair':'source #'+o.source_index}`;
 const t=state.data.iou_threshold,methods=['gt','baseline','detzero'];const card=(name,value)=>`<div class="iou-card"><span>${name} IoU 3D</span><strong class="${value==null?'muted':value>=t?'good':'bad'}">${fmt(value)}</strong><small>${value==null?'Chưa có cặp GT':value>=t?'Đạt':'Chưa đạt'}${value==null?'':` ngưỡng ${t.toFixed(2)} của cặp này`}</small></div>`;
 const field=(title,format)=>`<tr><td>${title}</td>${methods.map(m=>`<td>${records[m]?format(records[m]):'—'}</td>`).join('')}</tr>`;
 const yaw=r=>{const [w,x,y,z]=r.box.rotation;return Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))*180/Math.PI;};
 $('detailBody').innerHTML=`<div class="iou-cards">${card('BEVFusion',o.iou_baseline)}${card('DetZero',o.iou_detzero)}</div><table class="detail-table"><thead><tr><th>Thông số</th><th>GT</th><th>BEVFusion</th><th>DetZero</th></tr></thead><tbody>${field('Score',r=>r.score==null?'—':fmt(r.score))}${field('Width (m)',r=>fmt(r.box.size[0]))}${field('Length (m)',r=>fmt(r.box.size[1]))}${field('Height (m)',r=>fmt(r.box.size[2]))}${field('Center Z · global',r=>fmt(r.box.translation[2]))}${field('Bottom Z · global',r=>fmt(r.bottom_z))}${field('Δ bottom vs GT',r=>records.gt?fmt(r.bottom_z-records.gt.bottom_z):'—')}${field('Yaw (degree)',r=>fmt(yaw(r),1))}${field('Ego X / Y (m)',r=>`${fmt(r.center[0],2)} / ${fmt(r.center[1],2)}`)}</tbody></table><p class="detail-note">${o.gt?'GT '+esc(o.gt.slice(2)):'Prediction chưa ghép với GT trong phạm vi XY ≤ 0.5 m.'}<br>IoU ở đây được tính với GT đang chọn. Đạt ngưỡng hình học chưa đồng nghĩa TP nếu chưa xét class và ghép một-một theo confidence.</p>`;
}
async function selectObject(id){if(!state.data)return;state.selected=id;state.instance=selected()?.instance;renderList();renderDetail();resetSpatial();resetCamera();if(chosenCamera()!==state.cameraName)await loadCamera();requestDraw();}
function selectValid(preferred){const list=rows();let choice=list.find(o=>o.id===preferred);
 if(!choice&&$('follow').checked&&state.instance)choice=list.find(o=>o.instance===state.instance);
 choice??=list.find(o=>o.id===state.selected)??list.find(o=>o.source_index!=null&&o.gt!=null)??list[0];state.selected=choice?.id??null;state.instance=choice?.instance??null;}
async function loadFrame(preferred=null){if(!state.meta)return;const frame=Number($('frameNumber').value),score=Number($('score').value),iou=Number($('iou').value);if(!Number.isFinite(frame)||!Number.isFinite(score)||score<0||score>1){message('Frame hoặc score không hợp lệ.',true);return;}
 state.frame=Math.max(0,Math.min(state.meta.frames-1,Math.round(frame)));$('frameNumber').value=state.frame;$('frameSlider').value=state.frame;
 const version=++state.version;state.abort?.abort();state.abort=new AbortController();message(`Đang đọc frame ${state.frame}…`);$('export').disabled=true;state.points=new Float32Array();
 try{const response=await fetch(`/api/frame/${state.frame}?score=${score}&iou=${iou}&other=${$('other').checked?1:0}`,{signal:state.abort.signal});const data=await response.json();if(!response.ok)throw new Error(data.error||'Không đọc được frame');if(version!==state.version)return;
  state.data=data;selectValid(preferred);renderList();renderDetail();resetSpatial();resetCamera();await loadCamera();if(version!==state.version)return;$('export').disabled=false;message('');
  fetch(`/api/points/${state.frame}?limit=16000`).then(async response=>{if(!response.ok){const e=await response.json();throw new Error(e.error);}return response.arrayBuffer();}).then(buffer=>{if(version===state.version){state.points=new Float32Array(buffer);requestDraw();}}).catch(error=>{if(version===state.version)message(`Ảnh và box đã tải; LiDAR: ${error.message}`,true);});
 }catch(error){if(error.name!=='AbortError'&&version===state.version)message(error.message,true);}}
function refreshFilters(){selectValid(state.selected);renderList();renderDetail();resetSpatial();resetCamera();if(chosenCamera()!==state.cameraName)loadCamera();requestDraw();}
function step(amount){if(!state.meta)return;$('frameNumber').value=Math.max(0,Math.min(state.meta.frames-1,state.frame+amount));loadFrame();}
function stopPlay(){clearInterval(playTimer);playTimer=null;$('play').textContent='▶';$('play').setAttribute('aria-label','Phát các frame');}
function setMode(focus){state.focus=focus;for(const [id,active] of [['sceneMode',!focus],['objectMode',focus]]){$(id).classList.toggle('active',active);$(id).setAttribute('aria-pressed',String(active));}resetCamera();resetSpatial();}
function attachPointer(id,move,click){const canvas=$(id);let start=null,last=null,moved=false;
 canvas.addEventListener('pointerdown',e=>{if(!state.data)return;canvas.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];start=last;moved=false;});
 canvas.addEventListener('pointermove',e=>{if(!last)return;const next=[e.clientX,e.clientY];if(Math.hypot(next[0]-start[0],next[1]-start[1])>4)moved=true;move(next[0]-last[0],next[1]-last[1]);last=next;requestDraw();});
 canvas.addEventListener('pointerup',e=>{if(last&&!moved&&click){const rect=canvas.getBoundingClientRect();click(e.clientX-rect.left,e.clientY-rect.top);}last=null;});
 canvas.addEventListener('pointercancel',()=>{last=null;});}
function objectForRecord(method,record){return method==='gt'?record.id:state.data.objects.find(o=>o.source_index===record.source_index)?.id;}
attachPointer('cameraCanvas',(dx,dy)=>{state.cam.pan[0]+=dx;state.cam.pan[1]+=dy;},(x,y)=>{if(!state.cam.fit||!state.cameraName)return;const fit=state.cam.fit,p=[(x-fit.x)/fit.scale,(y-fit.y)/fit.scale],cam=state.data.cameras[state.cameraName];let winner=null,score=Infinity;
 for(const [method,records] of Object.entries(visibleGroups()))for(const r of records){const b=projectedBounds(r,cam);if(!b||p[0]<b[0]||p[0]>b[2]||p[1]<b[1]||p[1]>b[3])continue;const area=(b[2]-b[0])*(b[3]-b[1]);if(area<score){score=area;winner=objectForRecord(method,r);}}
 if(winner)selectObject(winner);});
attachPointer('bevCanvas',(dx,dy)=>{const rect=$('bevCanvas').getBoundingClientRect(),scale=Math.min(rect.width,rect.height)/(2*state.bev.radius);state.bev.center[0]+=dy/scale;state.bev.center[1]+=dx/scale;},(x,y)=>{const rect=$('bevCanvas').getBoundingClientRect();let winner=null,distance=25;
 for(const [method,records] of Object.entries(visibleGroups()))for(const r of records){const p=bevProject(r.center,state.bev.center,state.bev.radius,rect.width,rect.height),d=Math.hypot(p[0]-x,p[1]-y);if(d<distance){distance=d;winner=objectForRecord(method,r);}}if(winner)selectObject(winner);});
attachPointer('threeCanvas',(dx,dy)=>{state.three.yaw-=dx*.008;state.three.pitch=Math.max(-.2,Math.min(1.45,state.three.pitch+dy*.006));});
for(const [id,zoom] of [['cameraCanvas',e=>state.cam.zoom=Math.max(.4,Math.min(12,state.cam.zoom*Math.exp(-e.deltaY*.001)))],['bevCanvas',e=>state.bev.radius=Math.max(1,Math.min(120,state.bev.radius*Math.exp(e.deltaY*.001)))],['threeCanvas',e=>state.three.distance=Math.max(2,Math.min(240,state.three.distance*Math.exp(e.deltaY*.001)))]])$(id).addEventListener('wheel',e=>{e.preventDefault();zoom(e);requestDraw();},{passive:false});
$('threeCanvas').addEventListener('keydown',e=>{if(!e.key.startsWith('Arrow'))return;e.preventDefault();e.stopPropagation();if(e.key==='ArrowLeft')state.three.yaw-=.12;if(e.key==='ArrowRight')state.three.yaw+=.12;if(e.key==='ArrowUp')state.three.pitch=Math.min(1.45,state.three.pitch+.1);if(e.key==='ArrowDown')state.three.pitch=Math.max(-.2,state.three.pitch-.1);requestDraw();});
$('objects').addEventListener('click',e=>{const row=e.target.closest('tr[data-id]');if(row)selectObject(row.dataset.id);});
$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('play').onclick=()=>{if(playTimer)return stopPlay();$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Dừng phát');playTimer=setInterval(()=>{if(state.frame>=state.meta.frames-1)return stopPlay();step(1);},1000);};
$('frameNumber').onchange=()=>loadFrame();$('frameSlider').onchange=()=>{$('frameNumber').value=$('frameSlider').value;loadFrame();};
$('camera').onchange=()=>loadCamera();for(const id of ['score','iou','other'])$(id).onchange=()=>{if(id==='other')$('classFilter').options[0].textContent=$('other').checked?'Tất cả class':'Car + motorcycle + pedestrian';loadFrame();};
for(const id of ['classFilter','objectFilter'])$(id).onchange=refreshFilters;$('search').oninput=refreshFilters;
for(const id of ['showGT','showBaseline','showDetzero','showPoints'])$(id).onchange=requestDraw;
$('sceneMode').onclick=()=>setMode(false);$('objectMode').onclick=()=>setMode(true);$('cameraReset').onclick=resetCamera;$('bevReset').onclick=resetSpatial;$('threeReset').onclick=resetSpatial;
$('example').onchange=()=>{if(!$('example').value)return;const example=state.meta.examples[Number($('example').value)];$('classFilter').value=example.class==='truck'?'car':example.class;$('objectFilter').value='all';$('search').value='';$('frameNumber').value=example.frame;$('camera').value='AUTO';setMode(true);loadFrame(example.gt_id);};
window.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement?.tagName)||document.activeElement?.id==='threeCanvas')return;if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}if(e.key==='ArrowRight'){e.preventDefault();step(1);}if(e.key===' '){e.preventDefault();$('play').click();}});
new ResizeObserver(requestDraw).observe(document.querySelector('.workspace'));
$('export').onclick=()=>{if(!state.data||!state.image)return;drawCamera();drawBev();drawThree();const canvas=document.createElement('canvas');canvas.width=1600;canvas.height=900;const ctx=canvas.getContext('2d');ctx.fillStyle='#0b1120';ctx.fillRect(0,0,1600,900);ctx.fillStyle='#edf3ff';ctx.font='bold 24px system-ui';ctx.fillText(`VF · Frame ${state.frame} · ${state.cameraName} · ${selected()?.class||'all objects'}`,25,38);
 ctx.font='16px system-ui';let x=25;for(const [method,label] of Object.entries(NAMES)){ctx.fillStyle=COLORS[method];ctx.fillText(label,x,73);x+=method==='baseline'?260:230;}
 ctx.drawImage($('cameraCanvas'),20,100,1000,600);ctx.drawImage($('bevCanvas'),1040,100,540,290);ctx.drawImage($('threeCanvas'),1040,410,540,290);
 const o=selected();ctx.fillStyle='#c7d6ec';ctx.font='18px ui-monospace';ctx.fillText(`IoU 3D: BEVFusion ${fmt(o?.iou_baseline)}  |  DetZero ${fmt(o?.iou_detzero)}  |  delta ${fmt(o?.delta_iou)}`,25,750);
 ctx.font='14px system-ui';ctx.fillText(`Run: ${state.meta.detzero_run} | display score >= ${state.data.score} | selected source #${o?.source_index??'none'}`,25,790);ctx.fillText('GT green solid · BEVFusion orange dashed · DetZero blue solid | GT bottom is a reference, not a fitted road plane.',25,826);
 const link=document.createElement('a');link.download=`vf_frame_${String(state.frame).padStart(3,'0')}_${o?.class||'scene'}.png`;link.href=canvas.toDataURL('image/png');link.click();};
async function init(){try{const response=await fetch('/api/meta');if(!response.ok)throw new Error('Không đọc được metadata');state.meta=await response.json();$('frameNumber').max=$('frameSlider').max=state.meta.frames-1;$('frameTotal').textContent=`/ ${state.meta.frames-1}`;
 for(const c of state.meta.cameras){const option=document.createElement('option');option.value=c;option.textContent=c;$('camera').append(option);}
 state.meta.examples.forEach((e,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${e.class} · frame ${String(e.frame).padStart(3,'0')}`;$('example').append(option);});
 for(const c of state.meta.classes.filter(c=>!['car','motorcycle','pedestrian'].includes(c))){const option=document.createElement('option');option.value=c;option.textContent=c;$('classFilter').append(option);}
 $('runInfo').textContent=`${state.meta.run} / ${state.meta.detzero_run} · ${state.meta.frames} frames`;
 const example=state.meta.examples[0];if(example){$('frameNumber').value=example.frame;$('classFilter').value=example.class;}await loadFrame(example?.gt_id);
 }catch(error){message(error.message,true);}}
init();
