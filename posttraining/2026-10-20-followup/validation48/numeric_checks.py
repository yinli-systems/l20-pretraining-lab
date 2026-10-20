"""Independent restricted arithmetic checks for training solution identities.

No optional LaTeX parser and no changes to the frozen evaluation grader. Only
numeric ASTs are evaluated; unsupported symbolic statements remain unchecked.
"""
import ast
import math
import re
from fractions import Fraction


def convert(text):
    text=text.strip().replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    text=re.sub(r'C_\{?(\d+)\}?\^\{?(\d+)\}?',r'comb(\1,\2)',text)
    text=text.replace('\\left','').replace('\\right','').replace('\\,','').replace('\\!','')
    text=text.replace('\\cdot','*').replace('\\times','*').replace('×','*').replace('÷','/')
    text=text.replace('\\pi','pi').replace('π','pi').replace('^','**')
    def group(s,i):
        while i<len(s) and s[i].isspace():i+=1
        if i>=len(s) or s[i]!='{':raise ValueError('Unsupported unbraced numeric LaTeX')
        depth=1;j=i+1
        while j<len(s) and depth:
            depth+=(s[j]=='{')-(s[j]=='}');j+=1
        if depth:raise ValueError('Unbalanced numeric LaTeX')
        return s[i+1:j-1],j
    result='';i=0
    while i<len(text):
        if text.startswith('\\frac',i):
            a,j=group(text,i+5);b,i=group(text,j);result+='(('+convert(a)+')/('+convert(b)+'))'
        elif text.startswith('\\sqrt',i):
            a,i=group(text,i+5);result+='sqrt('+convert(a)+')'
        else:result+=text[i];i+=1
    result=result.replace('{','(').replace('}',')')
    result=re.sub(r'(?<=\d)(?=sqrt|pi|\()', '*',result)
    result=re.sub(r'(?<=\))(?=\d|sqrt|pi|\()', '*',result)
    return result.strip()


def numeric(text):
    if len(text)>250:raise ValueError('Bounded numeric expression')
    tree=ast.parse(convert(text),mode='eval')
    if sum(1 for _ in ast.walk(tree))>100:raise ValueError('Bounded AST')
    def visit(n):
        if isinstance(n,ast.Expression):return visit(n.body)
        if isinstance(n,ast.Constant) and type(n.value) in [int,float]:return Fraction(str(n.value))
        if isinstance(n,ast.Name) and n.id=='pi':return math.pi
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            return -visit(n.operand) if isinstance(n.op,ast.USub) else visit(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
            if isinstance(n.op,ast.Pow) and abs(b)<=20:return a**b
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and not n.keywords:
            if n.func.id=='sqrt' and len(n.args)==1:return math.sqrt(visit(n.args[0]))
            if n.func.id=='comb' and len(n.args)==2:
                a,b=map(visit,n.args)
                if a.denominator==b.denominator==1 and 0<=b<=a<=100:return Fraction(math.comb(int(a),int(b)))
        raise ValueError('Unsupported numeric expression')
    value=visit(tree)
    if not math.isfinite(float(value)):raise ValueError('Non-finite numeric result')
    return value


def check_solution(text):
    regions=re.findall(r'(?s)\$\$(.*?)\$\$|\\\[(.*?)\\\]|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)',text)
    checked=0;bad=[]
    for values in regions:
        region=next((x for x in values if x),'')
        if '<' in region or '>' in region or '\\approx' in region:continue
        for clause in re.split(r'\\(?:Rightarrow|Leftrightarrow|implies|impliedby)',region):
            parts=clause.split('=')
            for left,right in zip(parts,parts[1:]):
                try:
                    a,b=numeric(left.strip().strip('&')),numeric(right.strip().strip('&'))
                    delta=abs(float(a-b));scale=max(1,abs(float(a)),abs(float(b)))
                except Exception:continue
                checked+=1
                if delta>1e-6*scale:bad.append(dict(left=left,right=right,left_value=float(a),right_value=float(b)))
    return dict(numeric_equalities_checked=checked,contradictions=bad)


def canaries():
    assert check_solution('$1+1=2$')['numeric_equalities_checked']==1
    fraction=check_solution('$\\frac{2}{3}+\\frac{1}{3}=1$')
    assert fraction['numeric_equalities_checked']==1 and not fraction['contradictions']
    assert check_solution('$C_{6}^{3}=15$')['contradictions']
    assert check_solution('$R=2.5\\times\\frac{6}{5}=3\\times2=6$')['contradictions']
    assert check_solution('$x=\\frac{2023}{4096}=\\frac{1}{2} \\Rightarrow y=2$')['contradictions']
    assert check_solution('$x+x=3x$')['numeric_equalities_checked']==0
    return dict(passed=True,parser='restricted Python AST/Fraction, sqrt/comb numeric only',
                optional_latex_parser_used=False,frozen_evaluation_unchanged=True)


if __name__=='__main__':print(canaries())
