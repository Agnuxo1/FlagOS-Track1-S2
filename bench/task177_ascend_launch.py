"""CUDA proxy verifies complete row coverage; actual Ascend rerun is still needed."""
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels.zimage_qk_rmsnorm_ascend import zimage_qk_rmsnorm


def main():
    torch.set_num_threads(1)
    cases = [(1, 1, 1, 128), (2, 31, 8, 128), (1, 262139, 1, 128),
             (1, 262140, 1, 128), (1, 262141, 1, 128), (1, 4096, 128, 128)]
    rows = []
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        for shape in cases:
            for layout in ['contiguous', 'fused_qkv_slice']:
                b, s, h, d = shape
                storage = torch.randn((b, s, h*3 if layout != 'contiguous' else h, d),
                                      device='cuda', dtype=torch.bfloat16)
                x = storage[:, :, h:2*h, :] if layout != 'contiguous' else storage
                w = torch.randn(128, device='cuda', dtype=torch.bfloat16)
                actual = zimage_qk_rmsnorm(x, w, 1e-6)
                # Independent Torch formula, chunked to bound verification memory.
                largest_error = 0.0
                for batch in range(b):
                    for start in range(0, s, 1024):
                        part = x[batch, start:start+1024].float()
                        expected = (part * torch.rsqrt(part.square().mean(-1, keepdim=True)+1e-6)).to(x.dtype)*w
                        selected = actual[batch, start:start+1024]
                        torch.testing.assert_close(selected, expected, rtol=.016, atol=1e-5)
                        largest_error = max(largest_error, float((selected.float()-expected.float()).abs().max()))
                assert actual.is_contiguous() and actual.data_ptr() != x.data_ptr()
                m = b*s*h
                rows.append(dict(seed=seed, shape=shape, layout=layout, rows=m,
                                 generic_grid=(m+3)//4, bounded_grid=min((m+3)//4,4096),
                                 passed=True, max_abs=largest_error))
                print(seed, shape, layout, 'passed', flush=True)
                del actual, x, storage, part, expected, selected
    out = Path('output'); out.mkdir(exist_ok=True)
    result = dict(gpu=torch.cuda.get_device_name(0), checks=rows,
                  all_passed=len(rows)==36, scope='Generated input numerical/coverage checks on RTX3090; no Ascend performance claim')
    (out/'ascend-launch-coverage.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
