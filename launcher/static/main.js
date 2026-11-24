'use strict';
const el=id=>document.getElementById(id);let selected=/^#[a-f0-9]{12}$/.test(location.hash)?location.hash.slice(1):null,latestRuns=[],ready=false,polling=false,uploading=false;
const signedOut='swm.signed-out',generation='swm.logout-generation';
const channel=typeof BroadcastChannel==='function'?new BroadcastChannel('swm-session'):null;
function leave(){document.body.classList.add('auth-check');location.replace('/ngrok/logout');}
function login(){document.body.classList.add('auth-check');sessionStorage.setItem('swm.login-pending','true');location.replace('/ngrok/login?redirect_path='+encodeURIComponent('/experiments/'));}
el('logout').onclick=()=>{localStorage.setItem(signedOut,'true');localStorage.setItem(generation,String(Date.now()+Math.random()));sessionStorage.removeItem('swm.login-pending');if(channel)channel.postMessage('logout');leave();};
if(channel)channel.onmessage=e=>{if(e.data==='logout')leave();};
window.addEventListener('storage',e=>{if(e.key===generation)leave();});
window.addEventListener('pageshow',e=>{if(e.persisted)location.reload();});
window.addEventListener('pagehide',()=>document.body.classList.add('auth-check'));
function error(message){el('error').textContent=message;el('error').hidden=!message;}
async function api(path,options={}){let r;try{r=await fetch('/experiments/api/'+path,{...options,credentials:'same-origin',cache:'no-store',redirect:'error',headers:{'Content-Type':'application/json'}});}catch(e){throw new Error('Connection interrupted. Reopen the lab if your session expired.');}if(r.status===401){login();throw new Error('Sign in again');}if(!r.ok)throw new Error((await r.text()).slice(0,400));return r.headers.get('Content-Type')?.includes('json')?r.json():r.text();}
function text(tag,value,cls){const n=document.createElement(tag);n.textContent=value;if(cls)n.className=cls;return n;}
function duration(run){const seconds=Math.max(0,Math.floor(((run.finished||Date.now()/1000)-(run.started||run.created))));return Math.floor(seconds/60)+'m '+seconds%60+'s';}
function renderRuns(runs){el('count').textContent=runs.length;el('runs').replaceChildren();if(!runs.length){el('runs').append(text('p','No runs yet. Choose a dataset and queue your first experiment.','empty'));return;}
 for(const run of runs){const box=text('article','','run'),top=text('div','','run-top'),name=text('button',run.name,'run-name');name.onclick=()=>selectRun(run.id);top.append(name,text('span',run.status,'state '+run.status));box.append(top);const meta=text('div','','run-meta');let progress=run.phase;if(run.status==='running'&&run.training_step&&run.phase==='training')progress+=' · '+run.training_step.toLocaleString()+'/'+run.steps.toLocaleString();if(['collecting validation incidents','collecting fresh forecast incidents'].includes(run.phase))progress+=' · '+(run.collected||0)+'/64';meta.append(text('span',progress+' · '+duration(run)));if(!['completed','failed','cancelled'].includes(run.status)){const cancel=text('button','Cancel','cancel');cancel.onclick=async()=>{try{cancel.disabled=true;await api('runs/'+run.id+'/cancel',{method:'POST',body:'{}'});await refresh();}catch(e){error(e.message);cancel.disabled=false;}};meta.append(cancel);}box.append(meta);box.append(text('p',(run.temporal_pilot?'TEMPORAL PILOT · no training · ':run.fresh_triage?'FRESH TRIAGE · frozen model · ':run.audit_source?'DIAGNOSTIC · frozen model · ':run.mode==='forecast'?'FORECAST · frozen model · ':run.mode==='test-only'?'TEST ONLY · ':'')+'Seed '+run.seed+' · '+run.steps.toLocaleString()+' steps'+(run.steps===1000?' · SMOKE TEST':'')+' · '+new Date(run.created*1000).toLocaleString(),'hint'));if(run.status==='running'){const bar=text('div','','bar'),fill=text('span','');fill.style.width=(run.phase==='training'?Math.min(60,60*(run.training_step||0)/run.steps):['collecting validation incidents','collecting fresh forecast incidents'].includes(run.phase)?65+30*(run.collected||0)/64:95)+'%';bar.append(fill);box.append(bar);}el('runs').append(box);}}
async function refreshDetail(){if(!selected)return;const r=await api('runs/'+selected);el('detail').hidden=false;el('detail-title').textContent=r.name;el('detail-meta').textContent=r.phase+' · model seed '+r.seed+' · test seed '+r.test_seed+' · '+r.version+(r.error?' · '+r.error:'');el('report').hidden=!['completed','failed','cancelled'].includes(r.status);el('report').href='/experiments/api/runs/'+r.id+'/report';el('metrics').replaceChildren();if(r.summary){const grid=text('div','','metrics');for(const [n,label] of [['latent6','Dreamer · 6 observations'],['latent8','Dreamer · 8 observations'],['history6','Telemetry · 6 observations'],['history8','Telemetry · 8 observations'],['baseline6','Previous model · same test · 6 observations']]){const m=r.summary.methods[n];if(!m)continue;const cell=text('div','','metric');cell.append(text('span',label),text('strong',m.correct+'/'+m.n),text('span','False pool diagnoses: '+r.summary.false_pool[n]));grid.append(cell);}el('metrics').append(grid);el('interpretation').textContent=(r.steps===1000?'Smoke test: this verifies the workflow, not model quality. ':'')+'Fresh test accuracy is not proof of generalization. All 8 windows were collected; 6-window scores do not measure real time savings. '+checks(r.summary);}else el('interpretation').textContent='Results appear after fresh validation. Serving inference is restored before the run is marked complete.';renderForecast(r);renderDiagnostic(r);renderTemporal(r);el('logs').textContent=await api('runs/'+r.id+'/logs');}
async function refresh(){if(polling)return;polling=true;try{latestRuns=await api('runs');renderRuns(latestRuns);renderTrend(latestRuns);if(!latestRuns.some(r=>r.id===selected))rememberRun((latestRuns.find(r=>r.status==='completed'&&(r.summary||r.forecast_summary||r.diagnostic_summary)&&r.steps!==1000)||latestRuns[0])?.id||null);if(selected)await refreshDetail();else el('detail').hidden=true;error('');}catch(e){error(e.message);}finally{polling=false;}}
el('run-form').onsubmit=async event=>{event.preventDefault();el('submit').disabled=true;try{const run=await api('runs',{method:'POST',body:JSON.stringify({name:el('name').value,dataset:el('dataset').value,seed:Number(el('seed').value),steps:Number(el('steps').value)})});rememberRun(run.id);await refresh();el('detail').scrollIntoView({behavior:'smooth',block:'nearest'});}catch(e){error(e.message);}finally{el('submit').disabled=!ready;}};
function datasetOptions(datasets,choose){const prior=choose||el('dataset').value;el('dataset').replaceChildren();for(const d of datasets){const option=text('option',d.name+' · '+d.incidents+' incidents'+(d.source==='uploaded'?' · uploaded':''));option.value=d.id;el('dataset').append(option);}if(datasets.some(d=>d.id===prior))el('dataset').value=prior;datasetDownload();}
function datasetDownload(){el('dataset-download').href='/experiments/api/datasets/'+encodeURIComponent(el('dataset').value)+'/download';}
el('dataset').onchange=datasetDownload;
el('upload-form').onsubmit=async event=>{event.preventDefault();if(uploading)return;uploading=true;el('upload-submit').disabled=true;el('upload-status').textContent='Reading and validating your dataset…';try{const file=el('upload-file').files[0];if(!file)throw new Error('Choose a JSONL training file.');if(file.size>2*1024*1024)throw new Error('The file must be no larger than 2 MiB.');const d=await api('datasets',{method:'POST',body:JSON.stringify({name:el('upload-name').value,jsonl:await file.text(),training_only:el('training-only').checked})});const cfg=await api('config');datasetOptions(cfg.datasets,d.id);el('upload-status').textContent='Accepted '+d.incidents+' incidents. Dataset selected for your next run. SHA-256: '+d.sha256;el('upload-file').value='';el('training-only').checked=false;error('');}catch(e){el('upload-status').textContent='Upload rejected: '+e.message;}finally{uploading=false;el('upload-submit').disabled=!ready;}};
async function boot(){try{if(localStorage.getItem(signedOut)==='true'&&sessionStorage.getItem('swm.login-pending')!=='true'){login();return;}const cfg=await api('config');localStorage.removeItem(signedOut);sessionStorage.removeItem('swm.login-pending');document.body.classList.remove('auth-check');el('identity').textContent=cfg.email+(cfg.admin?' · Admin':'');el('version').textContent='Code '+cfg.version;el('scope').textContent=cfg.admin?'Administrator view · all runs. You can cancel any run.':'The queue is shared. You can access and cancel your own runs.';datasetOptions(cfg.datasets);ready=true;el('submit').disabled=false;el('upload-submit').disabled=uploading;await refresh();}catch(e){error(e.message);}}
document.addEventListener('visibilitychange',()=>{if(!document.hidden)boot();});boot();setInterval(()=>{if(ready&&!document.hidden)refresh();},5000);

function rememberRun(id){selected=id;history.replaceState(null,'',location.pathname+location.search+(id?'#'+id:''));}
async function selectRun(id){rememberRun(id);try{await refreshDetail();el('detail').scrollIntoView({behavior:'smooth',block:'nearest'});}catch(e){error(e.message);}}
function checks(s){return 'Reliability: '+(s.reliable&&s.regime_gate?'passed':'not passed')+'. Matched improvement: '+(s.improvement_gate==null?'not evaluated':s.improvement_gate?'passed':'not passed')+'.';}
function fraction(m){return m?m.correct+'/'+m.n:'—';}
function renderTrend(runs){
 const eligible=runs.filter(r=>r.status==='completed'&&r.summary);
 const shown=eligible.filter(r=>el('include-smoke').checked||r.steps!==1000).sort((a,b)=>a.created-b.created);
 el('trend-body').replaceChildren();el('trend-chart').replaceChildren();
 el('trend-empty').textContent=shown.length?'Six-observation bars: Dreamer (purple), raw telemetry (teal).':eligible.length?'Only smoke-test results are available. Enable the checkbox to see them.':'No completed results yet. Active and failed runs remain listed above.';
 el('trend-table').hidden=!shown.length;
 for(const r of shown){
  const row=text('tr',''),identity=text('td',''),name=text('button',r.name,'run-name');name.onclick=()=>selectRun(r.id);
  identity.append(name,text('p',new Date(r.finished*1000).toLocaleString()+' · seed '+r.seed+' · '+r.steps.toLocaleString()+(r.mode==='test-only'?' original training steps · TEST ONLY':' steps')+(r.steps===1000?' · SMOKE TEST':'')+' · test seed '+r.test_seed,'hint'));row.append(identity);
  for(const method of ['latent6','history6','latent8','history8','baseline6'])row.append(text('td',fraction(r.summary.methods[method])));
  row.append(text('td',checks(r.summary)));
  const report=text('td',''),link=text('a','Download');link.href='/experiments/api/runs/'+r.id+'/report';report.append(link);row.append(report);el('trend-body').append(row);
  const group=text('div','','trend-run');group.append(text('strong',r.name));
  for(const [key,label] of [['latent6','Dreamer'],['history6','Telemetry']]){
   const m=r.summary.methods[key];if(!m||!m.n)continue;
   const line=text('div','','trend-line'),track=text('div','','trend-track'),bar=text('div','','trend-fill '+key);bar.style.width=(100*m.correct/m.n)+'%';track.append(bar);line.append(text('span',label),track,text('span',fraction(m)+' · '+(100*m.correct/m.n).toFixed(1)+'%'));group.append(line);
  }
  el('trend-chart').append(group);
 }
}
el('include-smoke').onchange=()=>renderTrend(latestRuns);

function renderForecast(r){
 if(r.mode!=='forecast')return;
 el('interpretation').textContent='Forecast experiment: frozen Dreamer predicts the next ledger measurement before seeing it. Controls are fitted on training data only.';
 if(!r.forecast_summary)return;
 const s=r.forecast_summary,grid=text('div','','metrics');el('metrics').replaceChildren();
 for(const [key,label] of [['dreamer','Dreamer prior'],['persistence','Persistence'],['matching_probe','Last matching probe'],['regression','Telemetry regression']]){
  const m=s.methods[key],cell=text('div','','metric');cell.append(text('span',label),text('strong',m.log_mae.toFixed(4)),text('span','Log MAE · '+m.milliseconds_mae.toFixed(1)+' ms MAE'));grid.append(cell);
 }
 el('metrics').append(grid);
 el('interpretation').textContent='Lower error is better. '+s.incidents+' incidents / '+s.predictions+' next-window predictions. Training-selected control: '+s.selected_control+'. Improvement gate: '+(s.improvement_gate?'passed':'not passed')+'. One-step prediction does not demonstrate unobserved-action planning.';
}

function renderDiagnostic(r){
 if(!r.audit_source&&!r.fresh_triage)return;
 el('interpretation').textContent=r.fresh_triage?'Fresh validation: fit and freeze readouts before collecting new incidents.':'Retrospective diagnostic on existing data; no new faults or training. Reconstruction has seen its target and is not a forecast baseline.';
 if(!r.diagnostic_summary)return;
 if(r.diagnostic_summary.kind==='triage'){
  el('interpretation').textContent=r.diagnostic_summary.prospective?'Fresh triage validation with frozen readouts. Six observations is the primary comparison; other prefixes are secondary. All windows were collected, so no measured early stopping.':'Retrospective triage comparison. Readouts selected on training data only; fresh validation required. All windows were collected; no measured early stopping.';
  const grid=text('div','','metrics');el('metrics').replaceChildren();
  for(const [prefix,methods] of Object.entries(r.diagnostic_summary.prefixes))for(const [name,m] of Object.entries(methods)){const cell=text('div','','metric');cell.append(text('span',prefix+' observations · '+name.replaceAll('_',' ')),text('strong',m.correct+'/'+m.n),text('span','False pool diagnoses: '+m.false_pool));grid.append(cell);}
  el('metrics').append(grid);return;
 }
 const grid=text('div','','metrics');el('metrics').replaceChildren();
 for(const [split,data] of Object.entries(r.diagnostic_summary.splits))for(const [method,m] of Object.entries(data.methods)){
  const cell=text('div','','metric');cell.append(text('span',split+' · '+method.replaceAll('_',' ')),text('strong',m.log_mae.toFixed(4)+' log MAE'),text('span',m.milliseconds_mae.toFixed(1)+' ms MAE · bias '+m.bias.toFixed(4)));grid.append(cell);
 }
 el('metrics').append(grid);
}

function renderTemporal(r){
 if(!r.temporal_pilot)return;
 el('interpretation').textContent='Mechanism feasibility pilot: checks retries, backlog and recovery. No model training or accuracy claim.';
 if(!r.temporal_summary)return;
 el('metrics').replaceChildren();
 if(r.temporal_summary.action_outcomes){
  const c=r.temporal_summary.action_outcomes;
  el('interpretation').textContent='Action-outcome screening: predictions saved before intervention. '+c.test_trials+' test trials across only '+c.independent_test_profiles+' independent workload profiles. No Dreamer comparison or reliable action-selection claim yet.';
  if(c.balanced)el('interpretation').textContent+=' Balanced timeout/load coverage; '+c.balanced.diagnostic_trials+' diagnostic repeats excluded from the test score.';
  if(c.validation)el('interpretation').textContent+=' Fresh validation of frozen predictors from '+c.validation.source_run+'; no refitting.';
  if(c.recovery)el('interpretation').textContent+=' Recovery preserved '+c.recovery.preserved_trials+' completed trials and the original frozen models.';
  for(const [name,m] of Object.entries(c.methods)){
   const cell=text('div','','metric');
   cell.append(text('span',name+(name===c.selected?' · selected using training data':'')),text('strong',(100*m.mae.error_fraction).toFixed(1)+' pp error-rate MAE'),text('span','Successful throughput MAE: '+m.mae.successful_requests_per_second.toFixed(2)+' req/s'),text('span','Latency MAE: '+(1000*m.mae.latency_seconds).toFixed(0)+' ms'),text('span','Queue MAE: '+m.mae.ledger_queue.toFixed(2)));
   el('metrics').append(cell);
  }
  el('metrics').append(text('p',c.all_restored?'All trial cleanup checks passed':'Check trial cleanup'));return;
 }
 if(r.temporal_summary.calibration){
  const c=r.temporal_summary.calibration;
  el('interpretation').textContent='Snapshot-only screening: a depth-two decision tree was frozen after 18 training incidents and tested on 18 separate incidents. No Dreamer training. Small sample; this is not confirmed generalization.';
  const cell=text('div','','metric');
  cell.append(text('span','Held-out snapshot diagnosis'),text('strong',c.correct+'/'+c.n),text('span','95% Wilson interval: '+c.wilson95.map(v=>(100*v).toFixed(0)+'%').join('–')));
  for(let i=0;i<c.labels.length;i++)cell.append(text('span',c.labels[i]+': '+c.confusion[i][i]+'/'+c.confusion[i].reduce((a,b)=>a+b,0)+' correct'));
  cell.append(text('span',c.snapshot_promising?'Simple snapshot method looks promising; validate more broadly.':'Screening threshold not reached; inspect errors before adding model complexity.'),text('span',c.all_restored?'All incident cleanup checks passed':'Check incident cleanup'));
  el('metrics').append(cell);return;
 }

 for(const row of r.temporal_summary.incidents){
  const cell=text('div','','metric');cell.append(text('span',row.mechanism+' · relief: '+row.relief),text('strong',row.restored?'Restored':'Check restoration'));
  for(const [name,stage] of Object.entries(row.stages))cell.append(text('span',name+': checkout '+(1000*stage.checkout.mean_window_latency).toFixed(0)+' ms · payment timeouts '+stage.payment.timeout_delta+' · ledger waiters '+stage.ledger.max_pool_waiters));
  el('metrics').append(cell);
 }
}
