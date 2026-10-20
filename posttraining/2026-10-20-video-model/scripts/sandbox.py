"""Execution worker: empty chroot, nobody UID, resource limits, syscall allowlist."""
import ctypes, errno, json, os, resource, sys
# Preload only standard-library modules commonly needed by benchmark solutions.
import math, collections, itertools, functools, heapq, bisect, re, string, typing, random, statistics, fractions, decimal, operator, copy, hashlib
# Load the OpenSSL digest implementation before the empty chroot is sealed.
hashlib.md5(b'preload').hexdigest()

def main():
    payload=json.loads(sys.stdin.read())
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.restype=ctypes.c_void_p; lib.seccomp_init.argtypes=[ctypes.c_uint32]
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]; lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p]
    jail=payload['jail']; os.makedirs(jail,exist_ok=True); os.chdir(jail); os.chroot(jail); os.chdir('/')
    os.setgroups([]); os.setgid(65534); os.setuid(65534)
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(3,3)); resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    resource.setrlimit(resource.RLIMIT_NPROC,(0,0)); resource.setrlimit(resource.RLIMIT_NOFILE,(16,16))
    ctx=lib.seccomp_init(0x00050000|errno.EPERM)
    if not ctx: raise RuntimeError('seccomp_init failed')
    for name in ['read','write','close','fstat','newfstatat','lseek','mmap','munmap','mprotect','brk','futex','clock_gettime','rt_sigaction','rt_sigprocmask','rt_sigreturn','sigaltstack','exit','exit_group','getpid','gettid','getrandom','sched_yield','madvise','prlimit64','getrusage']:
        num=lib.seccomp_syscall_resolve_name(name.encode())
        if num>=0 and lib.seccomp_rule_add(ctx,0x7fff0000,num,0)!=0: raise RuntimeError('seccomp rule failed')
    if lib.seccomp_load(ctx)!=0: raise RuntimeError('seccomp_load failed')
    try:
        ns={'__name__':'__main__'}
        exec(compile(payload['code'],'<solution>','exec'),ns)
        exec(compile(payload['test'],'<tests>','exec'),ns)
        if payload.get('entry_point'): ns['check'](ns[payload['entry_point']])
    except BaseException as exc:
        sys.stdout.write(json.dumps({'passed':False,'error':type(exc).__name__})+'\n'); return
    sys.stdout.write(json.dumps({'passed':True})+'\n')

if __name__=='__main__': main()
