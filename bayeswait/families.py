"""Infer task families and workload scale from shell commands in real agent traces.

describe_swe(cmd, repo)      : full command text (mini-SWE-agent trajectories)
describe_skeleton(sk, ...)   : TraceLab-style anonymised command skeleton (executables only)
Both return keys from most to least specific: exact (same task/session), l2 (same template, any task of the
repo/project), l3 (same runner in the repo/project), glob (same runner anywhere), plus runner and scale."""
import re, shlex, hashlib

def clean(cmd):
    c = (cmd or "").strip()
    c = re.sub(r"\s+\\\n\s*", " ", c)
    c = c.split("\n")[0] if not c.lstrip().startswith(("python -c", "python3 -c", "cat <<", "cat >")) else c
    c = re.sub(r"^\s*cd\s+\S+\s*(&&|;)\s*", "", c)
    c = re.sub(r"^\s*((\w+=\S+)\s+)+", "", c)                 # env-var prefixes
    c = re.sub(r"^\s*timeout\s+\S+\s+", "", c)
    c = re.sub(r"\s*(2>&1|2>/dev/null|>\s*/dev/null)\s*", " ", c)
    c = re.split(r"\s+\|\s+", c)[0]                             # drop pipes (head/tail/grep on output)
    return re.sub(r"\s+", " ", c).strip()

RUNNERS = [
    (r"^(python3?\s+)?(\./)?tests/runtests\.py\b", "django-runtests"),
    (r"^(python3?\s+-m\s+)?pytest\b", "pytest"),
    (r"^(python3?\s+)?(\./)?bin/test\b", "sympy-bintest"),
    (r"^python3?\s+-m\s+unittest\b", "unittest"),
    (r"^python3?\s+-c\b", "python-c"),
    (r"^python3?\s+\S+\.py\b", "python-script"),
    (r"^pip3?\s+install\b", "pip-install"),
    (r"^git\b", "git"),
    (r"^(grep|rg|find)\b", "search"),
]

def runner(c):
    for pat, name in RUNNERS:
        if re.search(pat, c): return name
    return (c.split() or ["?"])[0]

def targets(c, rn):
    """test targets (modules/files/node ids) and selection flags."""
    try: toks = shlex.split(c)
    except ValueError: toks = c.split()
    tg, flags, skip = [], [], False
    for i, t in enumerate(toks):
        if skip: skip = False; continue
        if t in ("-k", "-m", "--settings", "--parallel", "-p", "--tb", "-n") or (rn == "django-runtests" and t in ("-v", "--verbosity")):
            flags.append(t); skip = True; continue
        if t.startswith("-"): flags.append(t.split("=")[0]); continue
        if i == 0 or t in ("python", "python3", "pytest", "tests/runtests.py", "./tests/runtests.py", "bin/test", "./bin/test"): continue
        if t.isdigit(): continue
        tg.append(t)
    return tg, flags

def _describe(cmd, repo):
    """-> dict(exact, l1, l2, l3, scale) family keys from most to least specific."""
    c = clean(cmd); rn = runner(c)
    exact = c
    if rn == "python-c":                       # inline code: only exact repeats are comparable
        h = hashlib.md5(c.encode()).hexdigest()[:10]
        return dict(exact=exact, l1="pyc:" + h, l2=None, l3=repo + "|python-c", scale=None, runner=rn)
    if rn == "python-script":
        script = c.split()[1] if c.split()[0].startswith("python") else c.split()[0]
        return dict(exact=exact, l1="script:" + script, l2=None, l3=repo + "|python-script", scale=None, runner=rn)
    if rn in ("django-runtests", "pytest", "sympy-bintest", "unittest"):
        tg, flags = targets(c, rn)
        if rn == "django-runtests":            # test labels -> app modules
            units = sorted({t.split(".")[0] for t in tg})
        else:                                  # files (strip ::node ids)
            units = sorted({t.split("::")[0] for t in tg})
        sel = ("-k" in flags) or any("::" in t for t in tg) or (rn == "django-runtests" and any("." in t for t in tg))
        key = rn + ":" + ",".join(units) + (":sel" if sel else "")
        return dict(exact=exact, l1=key, l2=repo + "|" + key, l3=repo + "|" + rn, scale=max(len(tg), 1), runner=rn)
    return dict(exact=exact, l1=rn + ":" + c[:80], l2=None, l3=repo + "|" + rn, scale=None, runner=rn)


def describe_swe(cmd, repo, task):
    d = _describe(cmd, repo)
    return dict(exact=(task, d["exact"]), l2=d["l2"] or ("task-template", task, d["l1"]), l3=d["l3"],
                glob=d["runner"], runner=d["runner"], scale=d["scale"])

def describe_skeleton(sk, project, session):
    first = sk.split()[0] if sk.split() else "?"
    return dict(exact=(session, sk), l2=(project, sk), l3=(project, first), glob=sk, runner=first, scale=None)
