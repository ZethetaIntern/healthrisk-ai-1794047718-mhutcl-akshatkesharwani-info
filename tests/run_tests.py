"""Tiny runner for machines without pytest:  python tests/run_tests.py   (with pytest: pytest --cov)"""
import glob
import importlib.util
import os
import sys
import traceback

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.chdir(root)
passed = failed = 0
for path in sorted(glob.glob("tests/test_*.py")):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("test_"):
            try:
                getattr(mod, name)()
                passed += 1
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
