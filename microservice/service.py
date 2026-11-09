"""Real HTTP service chain with leased faults and measured telemetry.
Control port is separately isolated from the policy by NetworkPolicy.
"""
import collections, concurrent.futures, http.server, json, os, queue, threading, time, urllib.request
ROLE=os.environ['SERVICE_ROLE']
NEXT=os.environ.get('DOWNSTREAM','')
LOCK=threading.RLock()
ROWS=collections.deque(maxlen=4096)
DEFAULT={'cpu_seconds':0.001,'delay_seconds':0.0,'pool_size':8,'db_hold':0.04}
CONFIG=dict(DEFAULT);EXPIRES=0.;POOL=None

class Pool:
    def __init__(self,size):self.slots=threading.BoundedSemaphore(size);self.idle=queue.Queue()
    def query(self,hold):
        started=time.monotonic()
        if not self.slots.acquire(timeout=5):raise TimeoutError('pool acquisition timeout')
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
            if conn:self.idle.put(conn)
            self.slots.release()
    def close(self):
        while not self.idle.empty():self.idle.get_nowait().close()

def snapshot():
    with LOCK:rows=list(ROWS)
    def mean(k):return sum(r[k] for r in rows)/max(1,len(rows))
    def p95(k):
        x=sorted(r[k] for r in rows);return x[min(len(x)-1,int(len(x)*.95))] if x else 0.
    return {'service':ROLE,'count':len(rows),'timestamp':time.time(),
            'error_rate':mean('error'),'latency_p95':p95('elapsed'),
            'cpu_mean':mean('cpu'),'pool_wait_p95':p95('pool_wait'),
            'db_mean':mean('db'),'downstream_p95':p95('downstream'),
            'self_p95':p95('self_time')}

def apply(config):
    global CONFIG,POOL,EXPIRES
    allowed=set(DEFAULT)|{'lease_seconds'}
    if set(config)-allowed:raise ValueError('unsupported control field')
    new={**DEFAULT,**{k:v for k,v in config.items() if k in DEFAULT}}
    for k,high in [('cpu_seconds',.15),('delay_seconds',.6),('db_hold',.12)]:
        if not 0<=float(new[k])<=high:raise ValueError(k)
    if not isinstance(new['pool_size'],int) or not 1<=new['pool_size']<=8:raise ValueError('pool_size')
    lease=float(config.get('lease_seconds',45))
    if not 1<=lease<=90:raise ValueError('lease must be bounded')
    with LOCK:
        if ROLE=='ledger':
            old=POOL;POOL=Pool(new['pool_size'])
            if old:old.close()
        CONFIG=new;EXPIRES=time.monotonic()+lease;ROWS.clear()

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
        if self.control:return self.reply({'error':'not found'},404)
        if self.path=='/health':return self.reply({'ready':True})
        if self.path=='/metrics':return self.reply(snapshot())
        if self.path!='/request':return self.reply({'error':'not found'},404)
        started=time.monotonic();cpu=0.;downstream=0.;waited=0.;db=0.;err=0;trace=[]
        with LOCK:cfg=CONFIG.copy();pool=POOL
        try:
            cpu_start=time.thread_time()
            while time.thread_time()-cpu_start<cfg['cpu_seconds']:sum(i*i for i in range(100))
            cpu=time.thread_time()-cpu_start
            if cfg['delay_seconds']:time.sleep(cfg['delay_seconds'])
            if ROLE=='ledger':waited,db=pool.query(cfg['db_hold'])
            if NEXT:
                t=time.monotonic()
                with urllib.request.urlopen(NEXT+'/request',timeout=8) as r:trace=json.load(r)['trace']
                downstream=time.monotonic()-t
        except Exception as e:err=1
        elapsed=time.monotonic()-started
        row=dict(elapsed=elapsed,cpu=cpu,pool_wait=waited,db=db,downstream=downstream,self_time=max(0,elapsed-downstream),error=err)
        with LOCK:ROWS.append(row)
        trace.append({'service':ROLE,'duration':elapsed,'error':bool(err)})
        self.reply({'trace':trace},503 if err else 200)
    def do_POST(self):
        if not self.control or self.path!='/control':return self.reply({'error':'not found'},404)
        try:
            n=int(self.headers.get('Content-Length',0))
            if not 0<n<2048:raise ValueError('invalid body size')
            apply(json.loads(self.rfile.read(n)));self.reply({'accepted':True})
        except (ValueError,TypeError):self.reply({'error':'invalid bounded control'},400)

if __name__=='__main__':
    apply(DEFAULT);threading.Thread(target=expire,daemon=True).start()
    class Control(Handler):control=True
    threading.Thread(target=http.server.ThreadingHTTPServer(('0.0.0.0',9090),Control).serve_forever,daemon=True).start()
    http.server.ThreadingHTTPServer(('0.0.0.0',8080),Handler).serve_forever()
