const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const source = fs.readFileSync(require('node:path').join(__dirname, '../deploy/ngrok/session-plugin/main.js'),'utf8');
const shared = new Map(), tabs = [];
function tab(authenticated = true) {
 const events={}, docEvents={}, redirects=[], requests=[];
 const location={search:'',replace:x=>redirects.push(x)};
 const history={location,listen:fn=>events.route=fn};
 const document={hidden:false,documentElement:{style:{}},addEventListener:(n,fn)=>docEvents[n]=fn};
 const window={location,addEventListener:(n,fn)=>events[n]=fn,pluginLib:{React:{useEffect:fn=>fn()},ReactRouter:{useHistory:()=>history},registerAppBarAction:fn=>fn()}};
 const localStorage={getItem:k=>shared.get(k)||null,setItem:(k,v)=>{shared.set(k,v); for (const t of tabs) t.events.storage?.({key:k,newValue:v});}};
 const fetch=async(url,options)=>{requests.push(options);if(!authenticated)throw new Error('OAuth redirect');return {ok:true};};
 vm.runInNewContext(source,{window,document,localStorage,URLSearchParams,Date,Math,fetch,AbortController,setTimeout,clearTimeout});
 const t={events,docEvents,redirects,document,requests,setAuthenticated:v=>authenticated=v};tabs.push(t);return t;
}
const settle=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 const a=tab(),b=tab();await settle();
 assert.equal(a.document.documentElement.style.visibility,'');
 a.events.route({search:'?logout=true'});await settle();
 assert.deepEqual(a.redirects,['/ngrok/logout']);assert.deepEqual(b.redirects,['/ngrok/logout']);
 assert.equal(b.document.documentElement.style.visibility,'hidden');
 a.events.route({search:'?logout=true'});assert.equal(a.redirects.length,1);
 const c=tab();await settle();assert.equal(c.redirects.length,0);
 c.setAuthenticated(false);c.events.pagehide();c.events.pageshow({persisted:true});await settle();
 assert.match(c.redirects[0],/^\/ngrok\/login\?redirect_path=/);
 assert.equal(c.document.documentElement.style.visibility,'hidden');
 const d=tab(false);await settle();assert.match(d.redirects[0],/^\/ngrok\/login/);
 assert.equal(d.requests[0].redirect,'error');assert.equal(d.requests[0].cache,'no-store');
 const e=tab();await settle();e.setAuthenticated(false);e.docEvents.visibilitychange();await settle();assert.match(e.redirects[0],/^\/ngrok\/login/);
 console.log('PASS: cross-tab logout, duplicate suppression, fresh login, cached-shell and Back authentication, tab-focus revalidation');
})().catch(e=>{console.error(e);process.exitCode=1;});
