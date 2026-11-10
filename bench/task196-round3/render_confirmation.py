# Copyright 2026 Francisco Angulo de Lafuente
# SPDX-License-Identifier: Apache-2.0
"""Render the recorded confirmation values; no experiment or GPU execution."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
result=json.loads(args.result.read_text());assert result['selected']=='THIN_FLAT'
fig,ax=plt.subplots(figsize=(12,6.6));fig.subplots_adjust(left=.33,right=.98,top=.86,bottom=.29)
fig.set_facecolor('#f7f9fc');ax.set_facecolor('#f7f9fc');ys=np.arange(3)
arms=['CURRENT','THIN_FLAT','THIN_PACKED'];calls=[result['all_confirmation_arms'][a]['call_us'] for a in arms];graphs=[result['all_confirmation_arms'][a]['graph_us'] for a in arms]
ax.axvline(1,color='#7c899d',lw=1.2,ls='--')
ax.scatter(calls,ys-.12,color='#1768ac',s=75,label='Call timing: includes Python and allocation',zorder=3)
ax.scatter(graphs,ys+.12,color='#d57121',s=75,marker='s',label='Graph replay: mostly excludes host costs',zorder=3)
for metric,value,y,color in [('call_shape_cluster_95CI',calls[1],.88,'#1768ac'),('graph_shape_cluster_95CI',graphs[1],1.12,'#d57121')]:
 ci=result[metric];ax.errorbar(value,y,xerr=[[value-ci[0]],[ci[1]-value]],color=color,capsize=4,lw=1.6,zorder=4)
for arr,offset in [(calls,-.12),(graphs,.12)]:
 for value,y in zip(arr,ys):ax.annotate(f'{value:.4f}x',(value,y+offset),xytext=(8,0),textcoords='offset points',va='center',fontsize=10)
ax.set_yticks(ys,['Received control','THIN_FLAT: selected on development','THIN_PACKED: retained alternative']);ax.set_ylim(2.35,-.4)
ax.set_xlim(.975,1.215);ticks=[1,1.05,1.10,1.15,1.20];ax.set_xticks(ticks,[f'{v:.2f}x' for v in ticks])
ax.set_xlabel('Confirmation geometric mean speedup versus received control',fontsize=10,labelpad=12)
ax.grid(axis='x',alpha=.16);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=12,labelsize=10)
ax.legend(loc='upper right',frameon=False,fontsize=8)
fig.suptitle('Lower host-call cost; unchanged flat GPU arithmetic',x=.03,y=.96,ha='left',fontsize=17,fontweight='bold')
fig.text(.03,.065,'Local RTX 3090 | 10 confirmation shapes, both dtypes | paired rounds\nError bars: selected arm, shape-cluster 95% CI. Confirmation did not reselect a variant.\nCall improvement does not establish a faster GPU kernel or a seven-chip speedup.\nFrancisco Angulo de Lafuente | ORCID 0009-0001-1634-7063\nSource: results/task196-round3-20261110/RESULT.json',fontsize=8,color='#495770',linespacing=1.5)
fig.savefig(args.output,dpi=160,bbox_inches='tight');plt.close(fig)
print(json.dumps({'rendered':str(args.output),'source':str(args.result),'no_gpu':True}))
