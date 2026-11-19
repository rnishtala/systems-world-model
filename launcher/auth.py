"""Authenticate the internal hop from the loopback ingress proxy."""
import hashlib,hmac,time

def signature(key,method,path,email,stamp,body):
    payload='\n'.join((method,path,email,stamp,hashlib.sha256(body).hexdigest())).encode()
    return hmac.new(key,payload,hashlib.sha256).hexdigest()

def verify(key,method,path,email,stamp,sig,body,now=None):
    try:
        if abs((time.time() if now is None else now)-float(stamp))>30:return False
        return hmac.compare_digest(sig,signature(key,method,path,email,stamp,body))
    except (ValueError,TypeError):return False
