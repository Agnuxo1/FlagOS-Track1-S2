import os,sys,subprocess
from pathlib import Path
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['TRITON_CACHE_DIR']=str(Path('.triton_cache').resolve())
try:
    import pytest
except ImportError:
    dest=Path('.test_deps').resolve()
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--target',str(dest),'pytest==8.3.4'],check=True)
    os.environ['PYTHONPATH']=str(dest)+os.pathsep+os.environ.get('PYTHONPATH','')
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_kernels.py'],check=True)
subprocess.run([sys.executable,'bench/bench.py','--runs','3','--rounds','5','--out','results'],check=True)
