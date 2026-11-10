"""Validate the exact packaged public export on new GPU cases, never submit."""
import os
os.environ['OMP_NUM_THREADS']='2'
import hashlib,json,time
from pathlib import Path
import torch
from bias_gelu_tanh import bias_gelu_tanh

root=Path(__file__).resolve().parent;pin=json.loads((root/'EXPECTED.json').read_text())
assert hashlib.sha256((root/'bias_gelu_tanh.py').read_bytes()).hexdigest()==pin['candidate_sha256']
assert torch.cuda.is_available() and torch.cuda.device_count()==1
torch.set_num_threads(2);start=time.monotonic();records=[]
for dtype in [torch.float16,torch.bfloat16]:
 for idx,shape in enumerate([[5,72],[3,2,2056],[524289,8]]):
    gen=torch.Generator().manual_seed(19620261113+idx)
    x=torch.randn(shape,generator=gen,dtype=torch.float32).to(device='cuda',dtype=dtype)
    bias=x.reshape(-1,shape[-1])[0] if idx==0 else torch.linspace(-1,1,shape[-1],device='cuda',dtype=dtype)
    x0=x.clone();b0=bias.clone()
    expected=torch.nn.functional.gelu((x.float()+bias.float()).to(dtype),approximate='tanh')
    result=bias_gelu_tanh(x,bias);torch.cuda.synchronize()
    tol=.01 if dtype==torch.float16 else .015
    torch.testing.assert_close(result,expected,atol=tol,rtol=tol)
    assert result.shape==x.shape and result.device==x.device and result.dtype==dtype
    assert result.untyped_storage().data_ptr() not in [x.untyped_storage().data_ptr(),bias.untyped_storage().data_ptr()]
    assert torch.equal(x,x0) and torch.equal(bias,b0)
    records.append({'shape':shape,'dtype':str(dtype),'bias_aliases_input':idx==0,'passed':True,'max_abs':float((result.float()-expected.float()).abs().max())})
    assert time.monotonic()-start<120
result={'kind':'exact_selected_public_export_GPU_smoke','job_directory':root.parent.name,'candidate_sha256':pin['candidate_sha256'],'selected':pin['selected'],'checks':records,'all_passed':True,'managed_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'elapsed_seconds':time.monotonic()-start,'official_seven_chip_validation':False}
(root/'SMOKE_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
