"""Archive-only transport recovery using the machine's configured HTTPS proxy.

Retains the original archive/member checks and science evidence unchanged.
No credential is printed and signed asset redirects still use fresh requests.
"""
from pathlib import Path
import sys,urllib.request

ROOT=Path(__file__).resolve().parents[3]
original_build_opener=urllib.request.build_opener


def configured_proxy_opener(*handlers):
    handlers=tuple(urllib.request.ProxyHandler() if isinstance(h,urllib.request.ProxyHandler) and not h.proxies else h for h in handlers)
    return original_build_opener(*handlers)


if __name__=='__main__':
    assert sys.argv[1:]==['T371']
    urllib.request.build_opener=configured_proxy_opener
    sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'))
    import small_parts
    small_parts.main('T371')
