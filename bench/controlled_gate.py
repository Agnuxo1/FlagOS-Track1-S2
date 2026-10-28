"""Three actual tiling interventions, correctness gated; NOT the official 8-chip score."""
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='2'
import csv,json,random,statistics,time,datetime
from pathlib import Path
import torch,triton,triton.language as tl

@triton.jit
def gate_mul(X,G,Y,D:tl.constexpr,B:tl.constexpr):
 row=tl.program_id(0);tile=tl.program_id(1)
 j=tile*B+tl.arange(0,B)
 g=tl.load(G+row).to(tl.float32)
 x=tl.load(X+row*D+j,j<D,0).to(tl.float32)
 tl.store(Y+row*D+j,x*tl.sigmoid(g),j<D)

SHAPES=[(1,128),(1,4096),(4,4096),(8,4096),(32,8192),(16,16384)]
ROUNDS=24;CALLS=100;SESSIONS=3

def main():
 assert torch.cuda.is_available(),'CUDA required'
 out=Path('output');out.mkdir(exist_ok=True)
 torch.manual_seed(20261028);rng=random.Random(20261028)
 records=[];checks=[]
 for session in range(SESSIONS):
  for n,d in SHAPES:
   x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16);g=torch.randn(n,1,device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
   ref=(x.float()*torch.sigmoid(g.float())).to(x.dtype)
   baseline_w=max(4,min(triton.next_power_of_2(triton.cdiv(d,256)),32))
   configs={'baseline':(triton.next_power_of_2(d),baseline_w),'F1_tile512':(512,4),'F2_tile1024':(1024,4),'F3_tile2048':(2048,8)}
   funcs={}
   for name,(block,warps) in configs.items():
    def f(b=block,w=warps):gate_mul[(n,triton.cdiv(d,b))](x,g,y,d,b,num_warps=w)
    f();torch.cuda.synchronize();torch.testing.assert_close(y,ref,rtol=1e-2,atol=2e-2)
    checks.append(dict(session=session,n=n,d=d,method=name,passed=True,max_abs=float((y.float()-ref.float()).abs().max())))
    for _ in range(100):f()
    funcs[name]=f
   torch.cuda.synchronize()
   for r in range(ROUNDS):
    order=list(funcs);rng.shuffle(order)
    for name in order:
     s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
     s.record()
     for _ in range(CALLS):funcs[name]()
     e.record();e.synchronize()
     records.append(dict(session=session,n=n,d=d,round=r,method=name,microseconds=s.elapsed_time(e)*1000/CALLS))
   print(json.dumps({'session':session,'shape':[n,d],'correctness':'passed'}),flush=True)
 with (out/'flagos_gate_raw.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 summary=[]
 for session in range(SESSIONS):
  for n,d in SHAPES:
   vals={m:[r['microseconds'] for r in records if r['session']==session and r['n']==n and r['d']==d and r['method']==m] for m in configs}
   for name,v in vals.items():summary.append(dict(session=session,n=n,d=d,method=name,median_us=statistics.median(v),speedup=statistics.median(vals['baseline'])/statistics.median(v)))
 # Report every shape, every session. No filtering of negative gains.
 result={'kind':'new_GPU_tiling_interventions_replication','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'triton':triton.__version__,'seed':20261028,'sessions':SESSIONS,'rounds':ROUNDS,'calls_per_sample':CALLS,'shapes':SHAPES,'summary':summary,'correctness':checks,'arithmetic_mean_speedup':{name:statistics.mean(r['speedup'] for r in summary if r['method']==name) for name in configs},'limitations':['One NVIDIA GPU, not the official eight-chip evaluator.','CUDA event timer includes launch pacing; output allocation excluded equally in every arm.','Three tile interventions, sessions are replication not separate experiments.','No advantage assumed before measuring.'],'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (out/'flagos_gate_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({'arithmetic_mean_speedup':result['arithmetic_mean_speedup'],'correctness_checks':len(checks)}),flush=True)
if __name__=='__main__':main()
