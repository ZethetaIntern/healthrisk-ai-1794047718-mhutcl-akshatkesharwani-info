"""Rough line coverage with the standard library only (pytest-cov was not installed). Use `pytest --cov` for the official number."""
import glob
import importlib.util
import os
import sys
import trace

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
os.chdir(root)
PKGS = ["data", "models", "financial", "simulation", "explainability"]
prefixes = tuple(os.path.join(root, p) + os.sep for p in PKGS)
hit = {}


def tracer(frame, event, arg):
    fn = frame.f_code.co_filename
    if not fn.startswith(prefixes):
        return None
    lines = hit.setdefault(fn, set())

    def local(frame, event, arg):
        if event == "line":
            lines.add(frame.f_lineno)
        return local
    lines.add(frame.f_lineno)
    return local


sys.settrace(tracer)
for path in sorted(glob.glob("tests/test_*.py")):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("test_"):
            getattr(mod, name)()
sys.settrace(None)
tot_exec = tot_hit = 0
print(f"{'file':52s} {'covered':>8s} {'lines':>6s}  pct")
for p in PKGS:
    for f in sorted(glob.glob(os.path.join(root, p, "**", "*.py"), recursive=True)):
        ex = set(trace._find_executable_linenos(f))
        if not ex:
            continue
        h = hit.get(f, set()) & ex
        tot_exec += len(ex)
        tot_hit += len(h)
        print(f"{os.path.relpath(f, root):52s} {len(h):8d} {len(ex):6d}  {100 * len(h) / len(ex):5.1f}")
print(f"TOTAL {tot_hit}/{tot_exec} = {100 * tot_hit / tot_exec:.1f}%")
