"""Auditable exclusions against retained old prompts and Python AST families."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

JOB=Path('/workspace/video-model/validation-48h')
WORK=Path('/dev/shm/video-model-validation48')


def norm(text):return ' '.join(re.findall(r'\w+',text.lower()))
def digest(text):return hashlib.sha256(norm(text).encode()).hexdigest()
def family(text):return hashlib.sha256(re.sub(r'\d+','#',norm(text)).encode()).hexdigest()


def code_family(code):
    tree=ast.parse(code)
    local={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    local|={n.arg for n in ast.walk(tree) if isinstance(n,ast.arg)}
    local|={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    mapping={}
    def variable(name):
        if name not in mapping:mapping[name]='v'+str(len(mapping))
        return mapping[name]
    class Canonical(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in local:node.id=variable(node.id)
            return node
        def visit_arg(self,node):
            node.arg=variable(node.arg);return self.generic_visit(node)
        def visit_FunctionDef(self,node):
            node.name=variable(node.name)
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
            return self.generic_visit(node)
        visit_AsyncFunctionDef=visit_FunctionDef
        def visit_ClassDef(self,node):
            node.name=variable(node.name);return self.generic_visit(node)
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant(value='LITERAL:'+type(node.value).__name__),node)
    value=ast.dump(Canonical().visit(tree),include_attributes=False)
    return hashlib.sha256(value.encode()).hexdigest()


def code_function_families(code):
    """Also detect a held-out implementation embedded in a larger module.

    Keep code_family unchanged: existing proposals and their role hashes remain
    reproducible. These additional keys are conservative exclusion controls,
    not a proof that different algorithms are semantically unrelated.
    """
    tree=ast.parse(code)
    result={code_family(code)}
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            module=ast.Module(body=[copy.deepcopy(node)],type_ignores=[])
            result.add(code_family(ast.unparse(module)))
    return result


def build_fts(deadline):
    source=sqlite3.connect('file:'+str(WORK/'prior-prompts.sqlite')+'?mode=ro',uri=True)
    db=sqlite3.connect(WORK/'prior-fts.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS old USING fts5(normalized)')
    db.execute('CREATE TABLE IF NOT EXISTS cursor(n INTEGER)')
    old=db.execute('SELECT n FROM cursor').fetchone();start=old[0] if old else 0
    rows=source.execute('SELECT rowid,normalized FROM prompts WHERE rowid>? ORDER BY rowid',(start,))
    chunk=[];last=start
    for rowid,text in rows:
        if time.time()>deadline:raise TimeoutError('FTS build reached preparation bound')
        chunk.append((rowid,text));last=rowid
        if len(chunk)==8192:
            db.executemany('INSERT INTO old(rowid,normalized) VALUES(?,?)',chunk)
            db.execute('DELETE FROM cursor');db.execute('INSERT INTO cursor VALUES(?)',(last,));db.commit();chunk=[]
    if chunk:
        db.executemany('INSERT INTO old(rowid,normalized) VALUES(?,?)',chunk)
        db.execute('DELETE FROM cursor');db.execute('INSERT INTO cursor VALUES(?)',(last,));db.commit()
    count=db.execute('SELECT count(*) FROM old').fetchone()[0]
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();source.close()
    (JOB/'prior-fts-complete.json').write_text(json.dumps(dict(records=count,path=str(WORK/'prior-fts.sqlite'),screen='all contiguous normalized 13-word phrases; exact and numeric-family keys for short questions'),indent=2))


class PriorScreen:
    def __init__(self):
        self.db=sqlite3.connect('file:'+str(WORK/'prior-prompts.sqlite')+'?mode=ro',uri=True)
        self.fts=sqlite3.connect('file:'+str(WORK/'prior-fts.sqlite')+'?mode=ro',uri=True)
    def matches(self,text):
        if self.db.execute('SELECT 1 FROM prompts WHERE hash=? OR family_hash=? LIMIT 1',(digest(text),family(text))).fetchone():return True
        words=norm(text).split()
        phrases=['"'+' '.join(words[i:i+13])+'"' for i in range(max(0,len(words)-12))]
        for begin in range(0,len(phrases),64):
            if self.fts.execute('SELECT rowid FROM old WHERE old MATCH ? LIMIT 1',(' OR '.join(phrases[begin:begin+64]),)).fetchone():return True
        return False
