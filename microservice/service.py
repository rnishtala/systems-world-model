"""Real HTTP service chain with leased faults and measured telemetry.
Control port is separately isolated from the policy by NetworkPolicy.
"""
import collections, concurrent.futures, http.server, json, os, queue, threading, time, urllib.request, urllib.error, urllib.parse, socket, uuid
ROLE=os.environ['SERVICE_ROLE']
NEXT=os.environ.get('DOWNSTREAM','')
LOCK=threading.RLock()
ROWS=collections.deque(maxlen=4096)
DEFAULT={'cpu_seconds':0.001,'delay_seconds':0.0,'pool_size':8,'db_hold':0.04,'retry_limit':0,'attempt_timeout':8.0,'request_budget':8.0,'retry_backoff':0.0,'max_active':64}
COUNTERS=collections.Counter();SEQUENCE=0;BOOT_ID=uuid.uuid4().hex

def increment(name,amount=1):
    with LOCK:COUNTERS[name]+=amount

def downstream_call(url,cfg,fetch,clock=time.monotonic,sleep=time.sleep,record=increment):
    deadline=clock()+cfg['request_budget']
    for attempt in range(cfg['retry_limit']+1):
        remaining=deadline-clock()
        if remaining<=0:raise TimeoutError('downstream request budget exhausted')
        record('downstream_attempts')
        try:return fetch(url,min(cfg['attempt_timeout'],remaining))
        except (TimeoutError,socket.timeout,urllib.error.URLError) as exc:
            reason=getattr(exc,'reason',exc)
            timed_out=isinstance(reason,(TimeoutError,socket.timeout))
            retryable=timed_out or isinstance(exc,urllib.error.HTTPError) and exc.code in (502,503,504)
            if timed_out:record('downstream_timeouts')
            if not retryable or attempt>=cfg['retry_limit']:raise
            remaining=deadline-clock()
            if remaining<=cfg['retry_backoff']:raise TimeoutError('downstream request budget exhausted')
            if cfg['retry_backoff']:sleep(cfg['retry_backoff'])
            record('retries')
    raise RuntimeError('unreachable retry state')

def admit_request():
    """Atomic work admission: timed-out callers cannot grow server work unbounded."""
    with LOCK:
        COUNTERS['original_requests']+=1
        admitted=COUNTERS['active_requests']<CONFIG['max_active']
        if admitted:COUNTERS['active_requests']+=1
        else:COUNTERS['admission_rejections']+=1
        return admitted,CONFIG.copy(),POOL

def fetch_json(url,timeout):
    with urllib.request.urlopen(url,timeout=timeout) as response:return json.load(response)
CONFIG=dict(DEFAULT);EXPIRES=0.;POOL=None

class Pool:
    def __init__(self,size):self.slots=threading.BoundedSemaphore(size);self.idle=queue.Queue();self.retired=False
    def query(self,hold):
        started=time.monotonic()
        increment('pool_waiters')
        try:acquired=self.slots.acquire(timeout=5)
        finally:increment('pool_waiters',-1)
        if not acquired:raise TimeoutError('pool acquisition timeout')
        increment('pool_in_use')
        waited=time.monotonic()-started;conn=None
        try:
            try:conn=self.idle.get_nowait()
            except queue.Empty:
                import pg8000.dbapi
                conn=pg8000.dbapi.connect(host='swm-postgres',user='postgres',database='postgres',timeout=5)
                conn.autocommit=True
            begin=time.monotonic();cur=conn.cursor();cur.execute('SELECT pg_sleep(%s)',(hold,));cur.fetchall();cur.close()
            return waited,time.monotonic()-begin
        finally:
            if conn:
                with LOCK:
                    if self.retired:conn.close()
                    else:self.idle.put(conn)
            self.slots.release();increment('pool_in_use',-1)
    def close(self):
        with LOCK:
            self.retired=True
            while not self.idle.empty():self.idle.get_nowait().close()

def snapshot(after=None):
    with LOCK:
        allrows=list(ROWS);sequence=SEQUENCE;counters=dict(COUNTERS)
    rows=allrows if after is None else [r for r in allrows if r.get('sequence',0)>after]
    gap=after is not None and after<sequence and (not allrows or after<allrows[0].get('sequence',0)-1)
    def mean(k):return sum(r[k] for r in rows)/max(1,len(rows))
    def p95(k):
        x=sorted(r[k] for r in rows);return x[min(len(x)-1,int(len(x)*.95))] if x else 0.
    return {'service':ROLE,'count':len(rows),'timestamp':time.time(),'boot_id':BOOT_ID,
            'sequence':sequence,'window_gap':gap,'counters':counters,
            'error_rate':mean('error'),'latency_p95':p95('elapsed'),
            'cpu_mean':mean('cpu'),'pool_wait_p95':p95('pool_wait'),
            'db_mean':mean('db'),'db_query_count':sum(r['db']>0 for r in rows),
            'db_query_mean':sum(r['db'] for r in rows)/max(1,sum(r['db']>0 for r in rows)),
            'downstream_p95':p95('downstream'),
            'self_p95':p95('self_time')}

def reset_metrics():
    """Call only after all measurement requests have drained. Preserve fault lease/pool."""
    with LOCK:ROWS.clear()

def apply(config):
    global CONFIG,POOL,EXPIRES
    allowed=set(DEFAULT)|{'lease_seconds'}
    if set(config)-allowed:raise ValueError('unsupported control field')
    new={**DEFAULT,**{k:v for k,v in config.items() if k in DEFAULT}}
    for k,high in [('cpu_seconds',.15),('delay_seconds',.6),('db_hold',.12),('retry_backoff',.1)]:
        if not 0<=float(new[k])<=high:raise ValueError(k)
    if type(new['pool_size']) is not int or not 1<=new['pool_size']<=8:raise ValueError('pool_size')
    if type(new['max_active']) is not int or not 1<=new['max_active']<=64:raise ValueError('max_active')
    if type(new['retry_limit']) is not int or not 0<=new['retry_limit']<=2:raise ValueError('retry_limit')
    if not .05<=float(new['attempt_timeout'])<=8:raise ValueError('attempt_timeout')
    if not .1<=float(new['request_budget'])<=8:raise ValueError('request_budget')
    for key in ('cpu_seconds','delay_seconds','db_hold','retry_backoff','attempt_timeout','request_budget'):new[key]=float(new[key])
    lease=float(config.get('lease_seconds',45))
    if not 1<=lease<=90:raise ValueError('lease must be bounded')
    with LOCK:
        if ROLE=='ledger' and (POOL is None or new['pool_size']!=CONFIG['pool_size']):
            old=POOL;POOL=Pool(new['pool_size'])
            if old:old.close()
        CONFIG=new;EXPIRES=time.monotonic()+lease

def expire():
    while True:
        time.sleep(.5)
        with LOCK:expired=EXPIRES and time.monotonic()>EXPIRES
        if expired:apply({**DEFAULT,'lease_seconds':90})

class Handler(http.server.BaseHTTPRequestHandler):
    control=False
    def log_message(self,*args):pass
    def reply(self,data,status=200):
        body=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        global SEQUENCE
        if self.control:return self.reply({'error':'not found'},404)
        if self.path=='/health':return self.reply({'ready':True})
        if self.path=='/metrics':return self.reply(snapshot())
        if self.path.startswith('/metrics/window?'):
            try:
                query=urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query,strict_parsing=True)
                if set(query)!={'after'} or len(query['after'])!=1:raise ValueError('cursor')
                after=int(query['after'][0])
                if after<0 or after>SEQUENCE:raise ValueError('cursor')
                return self.reply(snapshot(after))
            except ValueError:return self.reply({'error':'invalid cursor'},400)
        if self.path!='/request':return self.reply({'error':'not found'},404)
        started=time.monotonic();cpu=0.;downstream=0.;waited=0.;db=0.;err=0;trace=[]
        admitted,cfg,pool=admit_request()
        try:
            if not admitted:raise RuntimeError('service admission limit')
            cpu_start=time.thread_time()
            while time.thread_time()-cpu_start<cfg['cpu_seconds']:sum(i*i for i in range(100))
            cpu=time.thread_time()-cpu_start
            if cfg['delay_seconds']:time.sleep(cfg['delay_seconds'])
            if ROLE=='ledger':waited,db=pool.query(cfg['db_hold'])
            if NEXT:
                t=time.monotonic()
                trace=downstream_call(NEXT+'/request',cfg,fetch_json)['trace']
                downstream=time.monotonic()-t
        except Exception as e:err=1
        elapsed=time.monotonic()-started
        row=dict(elapsed=elapsed,cpu=cpu,pool_wait=waited,db=db,downstream=downstream,self_time=max(0,elapsed-downstream),error=err)
        with LOCK:
            SEQUENCE+=1;row['sequence']=SEQUENCE;ROWS.append(row)
            COUNTERS['active_requests']-=int(admitted);COUNTERS['completed_requests']+=1
            COUNTERS['failed_requests']+=err
        trace.append({'service':ROLE,'duration':elapsed,'error':bool(err)})
        try:self.reply({'trace':trace},503 if err else 200)
        except (BrokenPipeError,ConnectionResetError):pass  # Timed-out callers do not cancel server work.
    def do_POST(self):
        if not self.control or self.path not in ('/control','/metrics/reset'):return self.reply({'error':'not found'},404)
        try:
            n=int(self.headers.get('Content-Length',0))
            if not 0<n<2048:raise ValueError('invalid body size')
            body=json.loads(self.rfile.read(n))
            if not isinstance(body,dict):raise ValueError('object required')
            if self.path=='/metrics/reset':
                if body:raise ValueError('empty object required')
                reset_metrics()
            else:apply(body)
            self.reply({'accepted':True})
        except (ValueError,TypeError):self.reply({'error':'invalid bounded control'},400)

if __name__=='__main__':
    apply(DEFAULT);threading.Thread(target=expire,daemon=True).start()
    class Control(Handler):control=True
    threading.Thread(target=http.server.ThreadingHTTPServer(('0.0.0.0',9090),Control).serve_forever,daemon=True).start()
    http.server.ThreadingHTTPServer(('0.0.0.0',8080),Handler).serve_forever()
