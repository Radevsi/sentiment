// Coordinates stay in source row order. No network libraries, telemetry or CDN.
export function screenPoint(x, y, camera, width, height) {
  return [(x-camera.x)*camera.scale+width/2, height/2-(y-camera.y)*camera.scale];
}
export function worldPoint(x, y, camera, width, height) {
  return [camera.x+(x-width/2)/camera.scale, camera.y-(y-height/2)/camera.scale];
}
export function pickPoints(points, x, y, camera, width, height, radius=10) {
  const matches=[];
  for(let i=0;i<points.length/2;i++) {
    const [sx,sy]=screenPoint(points[2*i],points[2*i+1],camera,width,height);
    const d=(sx-x)**2+(sy-y)**2;
    if(d<=radius**2) matches.push({id:i,d});
  }
  matches.sort((a,b)=>a.d-b.d||a.id-b.id);
  return {total:matches.length, ids:matches.slice(0,20).map(p=>p.id)};
}
export function zoomAt(camera, x, y, factor, width, height) {
  const [wx,wy]=worldPoint(x,y,camera,width,height);
  const scale=Math.max(1e-6,Math.min(1e9,camera.scale*factor));
  return {x:wx-(x-width/2)/scale, y:wy+(y-height/2)/scale, scale};
}

async function start() {
  const $=id=>document.getElementById(id), canvas=$('map'), status=$('status');
  const fail=error=>{status.textContent=`Error: ${error.message}`;};
  async function get(url) {const r=await fetch(url); if(!r.ok) throw new Error((await r.json()).error||r.statusText);return r;}
  const info=await (await get('/api/map')).json();
  const bytes=await (await get('/points.f32')).arrayBuffer();
  if(bytes.byteLength!==info.rows*8) throw new Error('Coordinate count does not match metadata');
  const data=new DataView(bytes), points=new Float32Array(info.rows*2);
  for(let i=0;i<points.length;i++) {points[i]=data.getFloat32(i*4,true);if(!Number.isFinite(points[i]))throw new Error('Invalid coordinate');}
  $('summary').textContent=`All ${info.rows.toLocaleString()} contexts · cosine UMAP`;
  $('row-id').max=info.rows-1;
  $('identity').textContent=JSON.stringify({run:info.source.run_name,model:info.source.embeddings,projection:info.projection,signature:info.signature},null,2);
  const gl=canvas.getContext('webgl',{alpha:false,antialias:false});
  if(!gl) throw new Error('WebGL is unavailable. Enable browser hardware acceleration and reload.');
  function shader(type,source) {const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
  const program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec2 position; uniform vec2 center; uniform vec2 scale; uniform float size; void main(){gl_Position=vec4((position-center)*scale,0.0,1.0);gl_PointSize=size;}'));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,'precision mediump float; uniform vec4 color; void main(){if(distance(gl_PointCoord,vec2(0.5))>0.5)discard;gl_FragColor=color;}'));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,points,gl.STATIC_DRAW);
  const position=gl.getAttribLocation(program,'position');gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);
  const uniforms=Object.fromEntries(['center','scale','size','color'].map(n=>[n,gl.getUniformLocation(program,n)]));
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  let camera={x:0,y:0,scale:1}, selected=-1, width=1,height=1,frame=0,selectionRequest=0;
  function draw(){frame=0;const ratio=window.devicePixelRatio||1;gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.047,.078,.125,1);gl.clear(gl.COLOR_BUFFER_BIT);gl.uniform2f(uniforms.center,camera.x,camera.y);gl.uniform2f(uniforms.scale,2*camera.scale/width,2*camera.scale/height);gl.uniform1f(uniforms.size,Number($('size').value)*ratio);gl.uniform4f(uniforms.color,.32,.78,.77,.6);gl.drawArrays(gl.POINTS,0,info.rows);if(selected>=0){gl.uniform1f(uniforms.size,12*ratio);gl.uniform4f(uniforms.color,1,.73,.32,1);gl.drawArrays(gl.POINTS,selected,1);}}
  function render(){if(!frame)frame=requestAnimationFrame(draw);}
  function fit(){const [lo,hi]=info.bounds;camera={x:(lo[0]+hi[0])/2,y:(lo[1]+hi[1])/2,scale:.88*Math.min(width/Math.max(hi[0]-lo[0],.001),height/Math.max(hi[1]-lo[1],.001))};render();}
  function resize(){width=canvas.clientWidth;height=canvas.clientHeight;const ratio=window.devicePixelRatio||1;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);render();}
  resize();fit();new ResizeObserver(resize).observe(canvas);
  $('reset').onclick=fit;$('size').oninput=render;
  async function select(id,recenter=false){
    if(!Number.isInteger(id)||id<0||id>=info.rows)throw new Error('row_id is outside this map');
    const request=++selectionRequest;selected=id;render();
    const point=await (await get(`/api/point?id=${id}`)).json();if(request!==selectionRequest)return;
    $('context').textContent=point.text;$('row-id').value=id;$('details').replaceChildren();
    for(const [name,value] of [['row_id',point.row_id],['Summed match_count',point.total_match_count],['Years present',point.year_count],['First year',point.first_year],['Last year',point.last_year]]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=name;dd.textContent=name==='Summed match_count'?value.toLocaleString():String(value);$('details').append(dt,dd);}
    if(recenter){camera.x=points[id*2];camera.y=points[id*2+1];render();}
    status.textContent=`All ${info.rows.toLocaleString()} points loaded · selected row_id ${id}`;
  }
  function local(event){const rect=canvas.getBoundingClientRect();return [event.clientX-rect.left,event.clientY-rect.top];}
  let drag=null;
  canvas.onpointerdown=e=>{const [x,y]=local(e);drag={x,y,origin:{...camera},moved:false};canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(!drag)return;const [x,y]=local(e),dx=x-drag.x,dy=y-drag.y;if(Math.hypot(dx,dy)>3)drag.moved=true;camera.x=drag.origin.x-dx/camera.scale;camera.y=drag.origin.y+dy/camera.scale;render();};
  canvas.onpointerup=e=>{if(!drag)return;const clicked=!drag.moved;drag=null;if(!clicked)return;const [x,y]=local(e),found=pickPoints(points,x,y,camera,width,height);$('candidates').replaceChildren();if(!found.ids.length){status.textContent='No point within 10 pixels. Zoom or use row_id lookup.';return;}if(found.total>1){const label=document.createElement('p');label.textContent=`${found.total} points near this click; nearest ${found.ids.length} shown. Zoom or use row_id to resolve overlaps.`;$('candidates').append(label);for(const id of found.ids){const button=document.createElement('button');button.textContent=`Row ${id}`;button.onclick=()=>select(id).catch(fail);$('candidates').append(button);}}select(found.ids[0]).catch(fail);};
  canvas.onpointercancel=()=>{drag=null;};
  canvas.addEventListener('wheel',e=>{e.preventDefault();const [x,y]=local(e);const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?height:1);camera=zoomAt(camera,x,y,Math.exp(-Math.max(-500,Math.min(500,delta))*.001),width,height);render();},{passive:false});
  canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();fail(new Error('WebGL context lost. Reload to restore the map.'));});
  $('row-form').onsubmit=e=>{e.preventDefault();select(Number($('row-id').value),true).catch(fail);};
  let searchRequest=0;
  $('find-form').onsubmit=async e=>{e.preventDefault();const request=++searchRequest;try{const results=await (await get(`/api/find?text=${encodeURIComponent($('find-text').value)}`)).json();if(request!==searchRequest)return;$('results').replaceChildren();const label=document.createElement('p');label.textContent=`${results.length} matches (first 50 by row_id; literal search, not an embedding query)`;$('results').append(label);for(const result of results){const button=document.createElement('button');button.textContent=`${result.row_id} · ${result.text}`;button.onclick=()=>select(result.row_id,true).catch(fail);$('results').append(button);}}catch(error){fail(error);}};
  status.textContent=`All ${info.rows.toLocaleString()} points loaded · drag, zoom and select`;
}
if(typeof document!=='undefined')start().catch(error=>{document.getElementById('status').textContent=`Error: ${error.message}`;});
