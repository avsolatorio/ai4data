"""Checks that hold for a fresh checkout: every script parses, every Python file compiles, every module imports,
every `scripts/...` path named in the code and documentation exists, and no machine-specific path is committed."""
import importlib
import pkgutil
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.sh"))
PYFILES = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py"))
DOCS = [ROOT / "README.md", ROOT / "RECIPE.md", ROOT / "REPRODUCE.md", *(ROOT / "docs").glob("*.md"), ROOT / "app" / "README.md",
        ROOT / "scripts" / "README.md"]
MACHINE = re.compile(r"/data/wb536061|/home/jupyter-|venvs/relexp|venvs/vllm")
SCRIPT_REF = re.compile(r"\bscripts/([A-Za-z0-9_.-]+\.(?:sh|py))\b")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_shell_script_parses(path):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("path", PYFILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_python_file_compiles(path):
    compile(path.read_text(), str(path), "exec")


def test_every_module_imports():
    import relevance_explainer
    failed = []
    for m in pkgutil.walk_packages(relevance_explainer.__path__, "relevance_explainer."):
        try:
            importlib.import_module(m.name)
        except Exception as e:  # collect all failures in one report
            failed.append(f"{m.name}: {type(e).__name__}: {e}")
    assert not failed, "\n".join(failed)


def test_referenced_scripts_exist():
    missing = set()
    for f in [*SCRIPTS, *PYFILES, *[d for d in DOCS if d.exists()]]:
        for name in SCRIPT_REF.findall(f.read_text()):
            if not (ROOT / "scripts" / name).exists() and name != "local.env":
                missing.add(f"{f.relative_to(ROOT)} -> scripts/{name}")
    assert not missing, "\n".join(sorted(missing))


def test_every_script_is_in_the_index():
    index = (ROOT / "scripts" / "README.md").read_text()
    listed = set(re.findall(r"^\| `([^`]+)` \|", index, re.M))
    present = {p.name for p in (ROOT / "scripts").iterdir() if p.suffix in (".sh", ".py")} | {"local.env.example"}
    assert present - listed == set(), f"not in scripts/README.md: {sorted(present - listed)}"
    assert listed - present == set(), f"listed but missing: {sorted(listed - present)}"


def test_no_machine_specific_paths():
    hits = []
    for f in [*SCRIPTS, *PYFILES]:
        for i, line in enumerate(f.read_text().split("\n"), 1):
            if MACHINE.search(line):
                hits.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()[:90]}")
    assert not hits, "\n".join(hits)


def test_scripts_source_env_before_using_its_variables():
    bad = []
    for f in SCRIPTS:
        lines = f.read_text().split("\n")
        uses = [i for i, l in enumerate(lines) if "$RELEXP_" in l and not l.lstrip().startswith("#")]
        if uses and f.name != "env.sh":
            src = [i for i, l in enumerate(lines) if l.strip() == ". scripts/env.sh"]
            if not src or src[0] > uses[0]:
                bad.append(f.name)
    assert not bad, bad


def test_env_precedence(tmp_path):
    """The caller's environment overrides scripts/local.env, which overrides the defaults."""
    cmd = ('cd "$1" && . scripts/env.sh && echo "$RELEXP_PY|$RELEXP_LOGS"')
    r = subprocess.run(["bash", "-c", cmd, "_", str(ROOT)], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "RELEXP_PY": "/opt/py", "RELEXP_LOGS": str(tmp_path / "l")})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"/opt/py|{tmp_path / 'l'}"
