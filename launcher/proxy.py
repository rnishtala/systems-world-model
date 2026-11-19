"""Loopback-only ingress: preserve Headlamp and route signed experiment traffic."""
import asyncio,os,time
from pathlib import Path
import aiohttp
from aiohttp import web,WSMsgType
from launcher.auth import signature
from launcher.common import identity

HOP={'connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer','transfer-encoding','upgrade','content-length'}

async def main():
    key=Path('/auth/key').read_bytes();static=Path(__file__).parent/'static'
    session=aiohttp.ClientSession(auto_decompress=False,timeout=aiohttp.ClientTimeout(total=None,sock_connect=10))
    async def handler(request):
        if request.path=='/healthz':return web.Response(text='ok')
        try:email=identity(request.headers)
        except PermissionError:raise web.HTTPUnauthorized(text='Sign in with Google to continue')
        path=request.path
        if path in ('/experiments','/experiments/'):
            response=web.Response(body=(static/'index.html').read_bytes(),content_type='text/html')
        elif path in ('/experiments/main.js','/experiments/style.css'):
            response=web.Response(body=(static/path.rsplit('/',1)[1]).read_bytes(),content_type='text/javascript' if path.endswith('.js') else 'text/css')
        else:
            experiments=path.startswith('/experiments/api/')
            upstream=os.environ.get('EXPERIMENTS_API','http://swm-experiments:8090') if experiments else 'http://127.0.0.1:4466'
            route=request.raw_path[len('/experiments'):] if experiments else request.raw_path
            headers={k:v for k,v in request.headers.items() if k.lower() not in HOP|{'host','x-swm-signature','x-swm-timestamp'}}
            headers['Host']='127.0.0.1:4466' if not experiments else 'swm-experiments:8090'
            if request.headers.get('Upgrade','').lower()=='websocket':
                if experiments:raise web.HTTPBadRequest(text='WebSocket not supported')
                ws=web.WebSocketResponse();protocols=tuple(x.strip() for x in request.headers.get('Sec-WebSocket-Protocol','').split(',') if x.strip())
                for k in list(headers):
                    if k.lower().startswith('sec-websocket-'):del headers[k]
                async with session.ws_connect(upstream+route,headers=headers,protocols=protocols,autoping=True) as remote:
                    ws=web.WebSocketResponse(protocols=[remote.protocol] if remote.protocol else ());await ws.prepare(request)
                    async def forward(src,dst):
                        async for message in src:
                            if message.type==WSMsgType.TEXT:await dst.send_str(message.data)
                            elif message.type==WSMsgType.BINARY:await dst.send_bytes(message.data)
                            elif message.type in (WSMsgType.CLOSE,WSMsgType.ERROR):break
                        await dst.close()
                    await asyncio.gather(forward(ws,remote),forward(remote,ws));return ws
            body=await request.read()
            if experiments:
                stamp=str(time.time());headers['X-Swm-Timestamp']=stamp
                headers['X-Swm-Signature']=signature(key,request.method,route,email,stamp,body)
                headers['X-Swm-Authenticated-Email']=email
            async with session.request(request.method,upstream+route,headers=headers,data=body,allow_redirects=False) as remote:
                response=web.StreamResponse(status=remote.status,headers=[(k,v) for k,v in remote.headers.items() if k.lower() not in HOP])
                response.headers['Cache-Control']='no-store'
                response.headers['X-Content-Type-Options']='nosniff'
                await response.prepare(request)
                async for chunk in remote.content.iter_chunked(65536):await response.write(chunk)
                await response.write_eof()
                return response
        response.headers['Cache-Control']='no-store';response.headers['X-Content-Type-Options']='nosniff'
        if path.startswith('/experiments'):
            response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'"
        return response
    app=web.Application(client_max_size=2*1024*1024);app.router.add_route('*','/{path:.*}',handler)
    runner=web.AppRunner(app);await runner.setup();await web.TCPSite(runner,'127.0.0.1',4467).start()
    try:await asyncio.Event().wait()
    finally:await runner.cleanup();await session.close()
if __name__=='__main__':asyncio.run(main())
