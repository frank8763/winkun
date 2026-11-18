"""Two verified HTTPS transports, unchanged archive and per-member SHA gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t638_memory_ready_recovery'))
import fetch as PythonTLS


def client():
    t=PythonTLS.ORIGINAL_CLIENT()
    curl_verified=t.response
    def response(method,url,body=None,headers=None):
        try:return PythonTLS.response(method,url,body,headers)
        except RuntimeError:
            # Original verified curl/HTTP1 path has an independent TLS stack.
            # It retains allowlisted redirect handling and credentials in a
            # pipe, never on argv/disk. No --insecure or auth sent to CDN.
            return curl_verified(method,url,body,headers)
    t.response=response
    return t


if __name__=='__main__':
    PythonTLS.Old.S.client=client
    PythonTLS.Old.main()
