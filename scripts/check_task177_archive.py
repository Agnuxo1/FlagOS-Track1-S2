"""Check the actual portal's required operator-named top-level function."""
import ast
import sys
import zipfile
from pathlib import Path


def check(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        allowed = {'zimage_qk_rmsnorm.py', 'zimage_qk_rmsnorm_ascend.py'}
        if 'zimage_qk_rmsnorm.py' not in names or len(names) != len(set(names)) or set(names) - allowed:
            raise ValueError('Expected correctly named cross-chip/chip-specific Python files')
        for name in names:
            source = archive.read(name).decode('utf-8')
            tree = ast.parse(source)
            matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'zimage_qk_rmsnorm']
            if len(matches) != 1:
                raise ValueError('Portal requires def zimage_qk_rmsnorm, not solution/reference aliases')
            args = matches[0].args
            if [x.arg for x in args.args] != ['x', 'weight', 'eps'] or args.vararg or args.kwarg or args.kwonlyargs:
                raise ValueError('Incorrect exported interface')
            expected = Path(__file__).resolve().parents[1] / 'kernels' / name
            if source != expected.read_text(encoding='utf-8'):
                raise ValueError('Archive does not contain the current checked source')
    print('Verified operator-named export, signature, UTF8 and exact current source:', path)


if __name__ == '__main__':
    check(Path(sys.argv[1]))
