const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const source = fs.readFileSync(require('node:path').join(__dirname, '../deploy/ngrok/session-plugin/main.js'),'utf8');
const shared = new Map();
const tabs = [];
function tab() {
 const events={}, docEvents={}, redirects=[];
 const location={search:'',replace:x=>redirects.push(x),reload:()=>redirects.push('reload')};
 const history={location,listen:fn=>events.route=fn};
 const document={hidden:false,documentElement:{style:{}},addEventListener:(n,fn)=>docEvents[n]=fn};
 const window={location,addEventListener:(n,fn)=>events[n]=fn,pluginLib:{React:{useEffect:fn=>fn()},ReactRouter:{useHistory:()=>history},registerAppBarAction:fn=>fn()}};
 const localStorage={getItem:k=>shared.get(k)||null,setItem:(k,v)=>{shared.set(k,v); for (const t of tabs) t.events.storage?.({key:k,newValue:v});}};
 vm.runInNewContext(source,{window,document,localStorage,URLSearchParams,Date,Math});
 const t={events,docEvents,redirects,document}; tabs.push(t);return t;
}
const a=tab(),b=tab();a.events.route({search:'?logout=true'});
assert.deepEqual(a.redirects,['/ngrok/logout']);assert.deepEqual(b.redirects,['/ngrok/logout']);
assert.equal(b.document.documentElement.style.visibility,'hidden');
a.events.route({search:'?logout=true'});assert.equal(a.redirects.length,1);
const c=tab();assert.equal(c.redirects.length,0);c.events.pagehide();c.events.pageshow({persisted:true});assert.deepEqual(c.redirects,['reload']);
console.log('PASS: cross-tab logout, immediate data hiding, duplicate suppression, new-session access, Back-cache revalidation');
