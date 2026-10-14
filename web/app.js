const $ = id => document.getElementById(id);
let state = null, selected = 0, lastGood = 0, lastTime = 0, lastFrame = null;
let explosions = [], traces = [], history = [], lastUpdate = -1, checkpointSignature = '';
const names = ['The proving ground','Switchback school','Long way home','Four-corner chaos','The serpent','Final exam'];
const number = (v, digits=0) => Number.isFinite(v) ? v.toLocaleString('en-US',{maximumFractionDigits:digits}) : '—';

async function control(command, extra={}) {
  try {
    const response = await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,...extra})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Control failed');
  } catch(error) { $('status').textContent = error.message; }
}
$('pause').onclick = () => control(state?.paused ? 'resume' : 'pause');
$('newClass').onclick = () => {if (confirm('Start a new rookie class? The current class and checkpoints will be saved.')) control('new_class');};
$('pauseTop').onclick=$('pause').onclick;
$('newClassTop').onclick=$('newClass').onclick;
$('fullscreen').onclick = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
for (const id of ['driver','speed','heldout','stochastic']) $(id).onchange = () => {
  control('view',{driver:$('driver').value,speed:Number($('speed').value),heldout:$('heldout').checked,stochastic:$('stochastic').checked});
  if(id==='heldout'){selected=0;traces=[];explosions=[];}
};

function resize(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), box = canvas.getBoundingClientRect();
  if(canvas.width !== Math.round(box.width*dpr) || canvas.height !== Math.round(box.height*dpr)){
    canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);
  }
  return [canvas.width,canvas.height];
}
function trackRadius(theta,p){return p[0]+p[1]*Math.cos(p[2]*theta+p[5])+p[3]*Math.sin(p[4]*theta);}
function roadPath(ctx,p){ctx.beginPath();for(let i=0;i<=360;i++){const t=i/360*Math.PI*2,r=trackRadius(t,p),x=r*Math.cos(t),y=-r*Math.sin(t);i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.closePath();}
function drawArena(canvas,track,now,mini=false){
  const [w,h]=resize(canvas), ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#0a1310';ctx.fillRect(0,0,w,h);
  const frame=state?.frame;if(!frame?.tracks?.[track]) return;
  const p=frame.tracks[track],extent=p[0]+Math.abs(p[1])+Math.abs(p[3])+18;
  const scale=Math.min(w,h)/(extent*2.1),cx=w/2,cy=h/2+(mini?0:8);
  ctx.save();ctx.translate(cx,cy);ctx.scale(scale,scale);
  // A coordinate grid makes the volume of simultaneous activity legible.
  ctx.lineWidth=.2;ctx.strokeStyle='#172820';ctx.beginPath();
  for(let i=-100;i<=100;i+=10){ctx.moveTo(i,-100);ctx.lineTo(i,100);ctx.moveTo(-100,i);ctx.lineTo(100,i);}ctx.stroke();
  roadPath(ctx,p);ctx.lineWidth=p[6]*2+4;ctx.strokeStyle='#07100c';ctx.stroke();
  roadPath(ctx,p);ctx.lineWidth=p[6]*2+1.4;ctx.strokeStyle='#68846a';ctx.stroke();
  roadPath(ctx,p);ctx.lineWidth=p[6]*2;ctx.strokeStyle='#27332c';ctx.stroke();
  roadPath(ctx,p);ctx.lineWidth=.22;ctx.setLineDash([2,3]);ctx.strokeStyle='#60745a';ctx.stroke();ctx.setLineDash([]);
  // Start / finish line across the road at theta zero.
  const start=trackRadius(0,p);for(let i=0;i<8;i++){ctx.fillStyle=i%2?'#d1e0c5':'#172119';ctx.fillRect(start-p[6]+i*p[6]/4,-.7,p[6]/4,1.4);}
  if(frame.obstacles){for(const [x,y] of frame.obstacle_xy[track]){ctx.fillStyle='#c9814b';ctx.strokeStyle='#ffd0a1';ctx.lineWidth=.3;ctx.beginPath();ctx.arc(x,-y,1.7,0,Math.PI*2);ctx.fill();ctx.stroke();}}
  if(!mini){
    ctx.lineWidth=.22;
    for(let i=0;i<traces.length;i++){if(frame.track_ids[i]!==track||traces[i].length<2)continue;ctx.strokeStyle=i%frame.tracks.length===track?'#66dacf22':'#66dacf15';ctx.beginPath();traces[i].forEach(([x,y],j)=>j?ctx.lineTo(x,-y):ctx.moveTo(x,-y));ctx.stroke();}
  }
  const age=Math.max(0,Math.min(.12,(Date.now()/1000-frame.time)));
  const feature=frame.track_ids.indexOf(track);
  for(let i=0;i<frame.cars.length;i++){
    if(frame.track_ids[i]!==track)continue;
    const [x,y,yaw,speed]=frame.cars[i], extra=age*state.view.speed;
    ctx.save();ctx.translate(x+Math.cos(yaw)*speed*extra,-y-Math.sin(yaw)*speed*extra);ctx.rotate(-yaw);
    if(mini){ctx.fillStyle=i===feature?'#ddff83':'#65ddd299';ctx.fillRect(-1,-.65,2,1.3);}
    else{
      ctx.globalAlpha=i===feature?1:.68;
      if(i===feature){ctx.shadowColor='#ddff83';ctx.shadowBlur=10/scale;}
      ctx.fillStyle=i===feature?'#ddff83':(i%5===0?'#98bdf5':'#65ddd2');
      ctx.beginPath();ctx.roundRect(-2.0,-.9,4.0,1.8,.4);ctx.fill();ctx.shadowBlur=0;
      ctx.fillStyle='#10211f';ctx.fillRect(.1,-.65,.8,1.3);
      ctx.fillStyle='#e1f5d2';ctx.fillRect(1.75,-.75,.25,.35);ctx.fillRect(1.75,.4,.25,.35);
      ctx.fillStyle='#fa775d';ctx.fillRect(-1.95,-.65,.2,.3);ctx.fillRect(-1.95,.35,.2,.3);
    }
    ctx.restore();
  }
  for(const event of explosions){if(event.track!==track)continue;const t=(now-event.born)/1000;if(t>.8)continue;ctx.globalAlpha=(1-t/.8)*.85;ctx.strokeStyle='#ff816b';ctx.lineWidth=mini?.5:.35;ctx.beginPath();ctx.arc(event.x,-event.y,1+t*9,0,Math.PI*2);ctx.stroke();
    if(!mini){for(let j=0;j<6;j++){const a=j*Math.PI/3+event.id;ctx.beginPath();ctx.moveTo(event.x+Math.cos(a)*(2+t*7),-event.y+Math.sin(a)*(2+t*7));ctx.lineTo(event.x+Math.cos(a)*(3+t*11),-event.y+Math.sin(a)*(3+t*11));ctx.stroke();}}}
  ctx.globalAlpha=1;ctx.restore();
  if(!mini){
    ctx.fillStyle='#d4e6ca';ctx.font=`600 ${Math.round(w/70)}px ui-monospace,monospace`;ctx.textAlign='center';ctx.fillText(state.view.driver==='rookie'?'DAY ONE':'DRIVING SCHOOL',cx,cy-7);
    ctx.fillStyle='#647c6b';ctx.font=`${Math.round(w/95)}px ui-monospace,monospace`;ctx.fillText(`${frame.cars.filter((_,i)=>frame.track_ids[i]===track).length} INDEPENDENT DRIVERS`,cx,cy+14);
  }
}
function buildCircuits(tracks){
  const container=$('circuits');container.innerHTML='';
  tracks.forEach((p,i)=>{const button=document.createElement('button');button.className='circuit'+(selected===i?' active':'');button.innerHTML=`<canvas id="circuit${i}"></canvas><span>${String(i+1).padStart(2,'0')} / ${state.view.heldout?'Unseen course':names[i]}</span>`;button.onclick=()=>{selected=i;for(const [j,node] of [...container.children].entries())node.classList.toggle('active',j===i);};container.append(button);});
}
function drawChart(){
  const canvas=$('learningChart'),[w,h]=resize(canvas),ctx=canvas.getContext('2d');ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='#293d30';ctx.lineWidth=1;for(let j=0;j<3;j++){const y=h-j*h/2;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  if(history.length<2)return;
  ctx.beginPath();history.forEach((p,i)=>{const x=i/(history.length-1)*w,y=h-4-p.value/100*(h-8);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.strokeStyle='#ddff83';ctx.lineWidth=2;ctx.stroke();
}
function updateUI(){
  const m=state.metrics,f=state.frame,e=f.policy_evaluation||m.evaluation;
  $('status').textContent=m.error?`${m.status}: ${m.error}`:m.status;
  const complete=m.complete||m.status?.startsWith('Class complete');
  $('connection').textContent=complete?'DELL / CLASS COMPLETE':m.max_load?'DELL / MAX LOAD':'DELL / LIVE';
  $('loadBadge').hidden=!m.max_load;
  $('introTag').textContent=m.max_load?'FULL THROTTLE / CONTINUOUS LEARNING':'ONE MACHINE. THOUSANDS OF SECOND CHANCES.';
  $('populationLabel').textContent=complete?'CARS TRAINED IN PARALLEL':'CARS LEARNING IN PARALLEL';
  $('rateLabel').textContent=complete?'LAST TRAINING RATE / SECOND':'EXPERIENCES PER SECOND';
  $('population').textContent=number(m.envs);$('throughput').textContent=number(m.steps_per_second);
  $('throughputDetail').textContent=Number.isFinite(m.overall_steps_per_second)?`${number(m.overall_steps_per_second)} overall / s, including tests`:'Simulation + training; tests excluded';
  $('miles').textContent=number(m.miles,1);$('crashes').textContent=number(m.crashes);
  $('simHours').textContent=`${number(m.simulated_hours,1)} hours of collective driving`;
  $('laps').textContent=`${number(m.laps)} completed training laps`;
  $('device').textContent=`${m.device || 'GB300'} · GPU simulation + PPO`;
  $('pause').textContent=state.paused?'Resume learning':'Pause learning';
  if(complete)$('pause').textContent='Class complete';
  $('pause').disabled=complete;
  $('pauseTop').textContent=$('pause').textContent;
  $('pauseTop').disabled=complete;
  $('rookieSuccess').textContent=m.baseline?`${m.baseline.lap_success_pct}%`:'—';
  $('success').textContent=e?`${e.lap_success_pct}%`:'—';$('successBar').style.width=`${e?.lap_success_pct||0}%`;
  $('update').textContent=`Update ${number(m.update||0)}`;
  $('evaluationDetail').textContent=e?`${e.cars} drivers · fixed seed ${e.seed} · a full lap without crashing`:'Waiting for the first test';
  $('evalLesson').textContent=e?`Tested at update ${e.policy_update??f.policy_update??m.update??0}: ${e.obstacles?'road + barriers':'road only'} · average ${e.mean_distance_m} m before completion or failure.`:'Tests run periodically; the live replay is a separate sample.';
  $('lesson').textContent=f.obstacles?'LESSON 02 / AVOID THE BARRIERS':'LESSON 01 / STAY ON THE ROAD';
  $('spectators').textContent=f.count?`${f.count} live policy replays · ${number(m.envs)} GPU training worlds`:'Waiting for the first drivers…';
  $('replayStats').textContent=f.count?`Replay: ${number(f.crashes)} crashes / ${number(f.laps)} laps`:'Independent driving worlds';
  const gpu=state.hardware.gpus?.find(g=>g.name.includes('GB300'));
  $('util').textContent=gpu?`${gpu.utilization}%`:'—';$('utilBar').style.width=`${gpu?.utilization||0}%`;
  $('power').textContent=gpu?`${number(gpu.watts)} W`:'—';$('temperature').textContent=gpu?`${gpu.temperature} °C`:'—';
  $('memory').textContent=Number.isFinite(m.gpu_memory_gb)?`${number(m.gpu_memory_gb,2)} GB`:'—';
  $('loadProfile').textContent=m.max_load?`Continuous training · ${number(m.policy_parameters/1e6,1)}M-parameter driver`:'';
  $('session').textContent=`DELL / ${m.run || 'LOCAL'} / NO CLOUD`;
  if(m.update!==lastUpdate){
    if(m.update<lastUpdate)history=[];
    if(e)history.push({value:e.lap_success_pct,update:m.update});
    if(history.length>150)history.shift();lastUpdate=m.update;
  }
  if(state.history?.length)history=state.history.map(e=>({value:e.lap_success_pct,update:e.update}));
  const signature=state.checkpoints.map(c=>c.key).join('|');
  if(signature!==checkpointSignature){
    checkpointSignature=signature;
    const select=$('driver');select.innerHTML='<option value="latest">Latest learner</option><option value="rookie">Day one · total mayhem</option>';
    const checkpoints=state.checkpoints.filter(c=>c.key.startsWith(m.run+'/')&&(c.key.includes('/best')||c.key.includes('/lesson-')));
    for(const cp of checkpoints){
      const option=document.createElement('option');option.value=cp.key;option.textContent=`${cp.key.split('/')[1]} · ${cp.evaluation?.lap_success_pct ?? '—'}%${cp.obstacles?' · barriers':''}`;select.append(option);
    }
  }
  if(document.activeElement!==$('driver'))$('driver').value=state.view.driver;
  $('speed').value=String(state.view.speed);$('heldout').checked=state.view.heldout;$('stochastic').checked=state.view.stochastic;
}
async function poll(){
  try{
    const response=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(5000)});
    if(!response.ok)throw new Error('Server unavailable');
    const next=await response.json(), now=performance.now();
    const reset=!state||next.view.heldout!==state.view.heldout||next.view.driver!==state.view.driver||next.metrics.run!==state.metrics.run;
    state=next;lastGood=Date.now();$('connection').textContent='DELL / LIVE';$('connection').parentElement.classList.add('connected');
    if(reset){traces=[];explosions=[];lastFrame=null;selected=0;}
    if(state.frame?.cars && state.frame.time!==lastTime){
      for(const event of state.frame.events){if(!explosions.some(x=>x.id===event[0]))explosions.push({id:event[0],track:event[1],x:event[2],y:event[3],born:now});}
      state.frame.cars.forEach((c,i)=>{
        if(!traces[i])traces[i]=[];const t=traces[i],prev=t[t.length-1];
        if(prev&&Math.hypot(prev[0]-c[0],prev[1]-c[1])>8)t.length=0;
        t.push([c[0],c[1]]);if(t.length>24)t.shift();
      });lastTime=state.frame.time;
    }
    if(state.frame.tracks && (reset||$('circuits').children.length!==state.frame.tracks.length)) buildCircuits(state.frame.tracks);
    updateUI();
  }catch(error){$('connection').textContent='CONNECTION LOST';$('connection').parentElement.classList.remove('connected');$('status').textContent=lastGood?'Connection lost — waiting for the Dell':'Connecting to the driving school…';}
  setTimeout(poll,100);
}
function animate(now){
  explosions=explosions.filter(e=>now-e.born<800);
  if(state?.frame?.tracks){
    drawArena($('arena'),selected,now);
    state.frame.tracks.forEach((_,i)=>{const canvas=$(`circuit${i}`);if(canvas)drawArena(canvas,i,now,true);});
    $('trackName').textContent=`Circuit ${String(selected+1).padStart(2,'0')} / ${state.view.heldout?'The unseen exam':names[selected]}`;
  }else{resize($('arena'));}
  drawChart();requestAnimationFrame(animate);
}
poll();requestAnimationFrame(animate);
