"""One prospective layout experiment; exact incumbent, frozen choice, two timing metrics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import csv,datetime,gc,hashlib,json,math,random,statistics,time
from pathlib import Path
import numpy as np
import torch,triton
import bias_gelu_tanh as k

ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'PROTOCOL.json').read_text())
OUT=ROOT/'output';OUT.mkdir(exist_ok=False);START=time.monotonic()
RAW=[];CHECKS=[]
def deadline():
 if time.monotonic()-START>P['internal_wall_seconds']:raise TimeoutError('Fixed wall limit')
def put(name,value):(OUT/name).write_text(json.dumps(value,indent=2))
def emit(value):print(json.dumps(value),flush=True)
def reference(x,b):return torch.nn.functional.gelu((x.float()+b.float()).to(x.dtype),approximate='tanh')
def make(shape,dtype,seed,pattern='normal'):
 g=torch.Generator(device='cpu').manual_seed(seed)
 x=torch.randn(shape,generator=g);b=torch.randn(shape[-1],generator=g)
 if pattern=='zeros':x.zero_();b.zero_()
 elif pattern=='cancellation_rounding':x=torch.linspace(-8,8,x.numel()).reshape(shape);b=torch.linspace(1,-1,shape[-1])
 elif pattern=='large_finite':x=x.clamp(-2,2)*1000;b=b.clamp(-2,2)*1000
 elif pattern=='tails':x=torch.linspace(-12,12,x.numel()).reshape(shape);b.zero_()
 return x.to(device='cuda',dtype=dtype),b.to(device='cuda',dtype=dtype)
def prepare(part,case,shape,dtype_name,seed,pattern='normal'):
 deadline();x,b=make(shape,getattr(torch,dtype_name),seed,pattern)
 x0=x.clone();b0=b.clone();expected=reference(x,b)
 funcs={'CURRENT':lambda:k.bias_gelu_tanh(x,b)}
 funcs.update({a:(lambda arm=a:k.round2_entry(x,b,arm)) for a in P['arms'][1:]})
 for arm,f in funcs.items():
  y=f();torch.cuda.synchronize()
  assert y.shape==x.shape and y.dtype==x.dtype and y.device==x.device
  assert y.untyped_storage().data_ptr() not in (x.untyped_storage().data_ptr(),b.untyped_storage().data_ptr())
  assert torch.equal(x,x0) and torch.equal(b,b0)
  torch.testing.assert_close(y,expected,**P['tolerances'][dtype_name])
  CHECKS.append({'stage':'eager','part':part,'case':case,'shape':shape,'dtype':dtype_name,'pattern':pattern,'arm':arm,'passed':True,'max_abs':float((y.float()-expected.float()).abs().max())})
 return x,b,expected,funcs
def timed(f,calls):
 s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
 s.record();f();e.record();e.synchronize()
 return s.elapsed_time(e)*1000/calls
def measure(part,shapes,seed):
 rng=random.Random(P['order_seed']+(part=='confirmation'))
 for case,shape in enumerate(shapes):
  for di,dtype in enumerate(P['dtypes']):
   x,b,expected,funcs=prepare(part,case,shape,dtype,seed+case*101+di)
   graphs={}
   for arm,f in funcs.items():
    for _ in range(P['warmup']):f()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for _ in range(P['calls']):sink=f()
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(sink,expected,**P['tolerances'][dtype])
    CHECKS.append({'stage':'graph','part':part,'case':case,'shape':shape,'dtype':dtype,'arm':arm,'passed':True})
    graphs[arm]=(graph,sink)
   for session in range(P['sessions']):
    for graph,_ in graphs.values():
     for _ in range(3):graph.replay()
    torch.cuda.synchronize()
    for rnd in range(P['rounds']):
     deadline();order=P['arms'].copy();rng.shuffle(order)
     for arm in order:
      f=funcs[arm]
      def calls():
       for _ in range(P['calls']):f()
      call_us=timed(calls,P['calls']);graph_us=timed(graphs[arm][0].replay,P['calls'])
      assert math.isfinite(call_us) and math.isfinite(graph_us) and min(call_us,graph_us)>0
      RAW.append({'part':part,'case':case,'shape':str(shape),'dtype':dtype,'session':session,'round':rnd,'arm':arm,'call_us':call_us,'graph_us':graph_us})
   put('CHECKS.json',CHECKS)
   with (OUT/'raw.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=RAW[0].keys());w.writeheader();w.writerows(RAW)
   put('PROGRESS.json',{'part':part,'case':case,'dtype':dtype,'rows':len(RAW),'checks':len(CHECKS)})
   emit({'event':'case_complete','part':part,'case':case,'dtype':dtype,'elapsed_seconds':time.monotonic()-START})
   del graphs,funcs,x,b,expected,sink,graph,f;gc.collect();torch.cuda.empty_cache()
def ratios(part,arm,metric):
 out=[]
 for case,dtype in sorted({(r['case'],r['dtype']) for r in RAW if r['part']==part}):
  rows=[r for r in RAW if r['part']==part and r['case']==case and r['dtype']==dtype]
  maps={a:{(r['session'],r['round']):r[metric] for r in rows if r['arm']==a} for a in {arm,'CURRENT'}}
  keys=set(maps[arm]);assert len(keys)==P['sessions']*P['rounds'] and all(set(v)==keys for v in maps.values())
  out.append({'case':case,'dtype':dtype,'ratio':statistics.median(maps['CURRENT'][i]/maps[arm][i] for i in sorted(keys))})
 return out
def gm(values):return math.exp(statistics.mean(math.log(x['ratio']) for x in values))
def main():
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 torch.set_num_threads(2)
 put('STARTED.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'managed_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'triton':triton.__version__,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file()},'protocol':P})
 for pi,pattern in enumerate(P['adversarial_panels']):
  for di,dtype in enumerate(P['dtypes']):prepare('adversarial',pi,[19,280],dtype,P['development_seed']+10000+pi*101+di,pattern)
 measure('development',P['development_shapes'],P['development_seed'])
 scores={a:{m:gm(ratios('development',a,m)) for m in ['call_us','graph_us']} for a in P['arms'][1:]}
 eligible=[a for a in P['arms'][1:] if scores[a]['graph_us']>=P['selection']['minimum_development_graph'] and scores[a]['call_us']>=P['selection']['minimum_development_call']]
 selected=max(eligible,key=lambda a:scores[a]['graph_us']) if eligible else 'CURRENT'
 selection={'selected':selected,'scores':scores,'rule':P['selection'],'before_confirmation':True}
 put('SELECTION_FROZEN.json',selection);frozen=hashlib.sha256((OUT/'SELECTION_FROZEN.json').read_bytes()).hexdigest();emit({'event':'selection_frozen',**selection})
 measure('confirmation',P['confirmation_shapes'],P['confirmation_seed'])
 assert hashlib.sha256((OUT/'SELECTION_FROZEN.json').read_bytes()).hexdigest()==frozen
 graph_ratios=ratios('confirmation',selected,'graph_us');call_ratios=ratios('confirmation',selected,'call_us')
 clusters=sorted({x['case'] for x in graph_ratios});logs=np.array([statistics.mean(math.log(x['ratio']) for x in graph_ratios if x['case']==c) for c in clusters])
 rng=np.random.default_rng(P['bootstrap_seed']);reps=np.exp(logs[rng.integers(0,len(logs),(10000,len(logs)))].mean(1));ci=np.quantile(reps,[.025,.975]).tolist()
 all_arms={a:{m:gm(ratios('confirmation',a,m)) for m in ['call_us','graph_us']} for a in P['arms']}
 offer=selected!='CURRENT' and ci[0]>P['confirmation']['offer_only_if_graph_CI_lower_gt'] and gm(call_ratios)>=P['confirmation']['offer_only_if_call_geomean_ge'] and all(c['passed'] for c in CHECKS)
 result={'selected':selected,'selection_sha256':frozen,'graph_geomean':gm(graph_ratios),'call_geomean':gm(call_ratios),'graph_shape_cluster_95CI':ci,'all_confirmation_arms':all_arms,'selected_graph_cases':graph_ratios,'selected_call_cases':call_ratios,'checks':len(CHECKS),'raw_rows':len(RAW),'all_correctness_passed':all(c['passed'] for c in CHECKS),'offer_one_official_followup':offer,'elapsed_seconds':time.monotonic()-START,'official_result':None,'limits':['One NVIDIA3090; no cross-chip claim.','Graph replay excludes most Python pacing and allocation; call timing retains pacing/allocation.','Fixed authored panel, no confirmation reranking; all negatives retained.']}
 put('RESULT.json',result);emit(result)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  put('FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.monotonic()-START});raise
