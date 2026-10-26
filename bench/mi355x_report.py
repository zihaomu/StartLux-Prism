"""Collect measured ceilings, latency and L2-Fabric counters; export a standalone figure."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='results/mi355x-perf')
    args=parser.parse_args()
    root=Path(args.directory)
    load=lambda name:json.loads((root/name).read_text())
    ceilings=load('roofline-baseline.json')
    latency=load('latency-ab.json')['rounds']
    names=['yesno','review1','readme3','startlux3','longpage2']
    qualified=load('qualified-baseline.json')
    summary=dict(gpu=ceilings['gpu'], model=load('latency-ab.json')['model'],
                 runtime={k:ceilings[k] for k in ('torch','hip','triton')},
                 kernel_identity=qualified['kernel_identity'],
                 engine_protocol=dict(prefix_cache=False,rounds=3,calls_per_round=30,warmups=3,alternating_modes=True),
                 ceilings=ceilings['ceilings'],theoretical=ceilings['theoretical'],engine={},counters={},profiles={})
    for name in names:
        modes={mode:statistics.median(r['median_ms'] for r in latency if r['suite']==name and r['mode']==mode)
               for mode in ('baseline','tuned')}
        summary['engine'][name]=dict(**modes,speedup=modes['baseline']/modes['tuned'],
                                     latency_reduction_percent=100*(1-modes['tuned']/modes['baseline']))
    for mode in ('baseline','tuned'):
        counters={}
        for path in (root/('counters-'+mode)).rglob('*counter_collection.csv'):
            for row in csv.DictReader(path.open()):
                counters[row['Counter_Name']]=dict(value=float(row['Counter_Value']),
                    kernel_us=(int(row['End_Timestamp'])-int(row['Start_Timestamp']))/1000)
        if counters:
            read=counters['FetchSize']
            write=counters['WRITE_SIZE']
            summary['counters'][mode]=dict(raw=counters,
                external_read_bytes=read['value']*1024, external_write_bytes=write['value']*1024,
                external_read_tb_s=read['value']*1024/read['kernel_us']/1e6,
                read_amplification_over_weight_bytes=read['value']*1024/(34816*5120),
                mfma_util_percent=counters['MfmaUtil']['value'])
        profile=load(mode+'-readme3.json')
        summary['profiles'][mode]=dict(graph_ms=profile['graph_ms'],eager_kernel_ms=profile['eager_kernel_ms'],
            gemm_ms=sum(r['ms'] for r in profile['kernels'] if r['name'].startswith('_w8a8_kernel')))
    if (root/'http-ab.json').exists():
        http=load('http-ab.json')['measurements']
        summary['http']={}
        for name in names:
            modes={mode:statistics.median(r['median_ms'] for r in http if r['suite']==name and r['mode']==mode)
                   for mode in ('baseline','tuned')}
            summary['http'][name]=dict(**modes,speedup=modes['baseline']/modes['tuned'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    fig,axes=plt.subplots(1,2,figsize=(13.5,4.6),layout='constrained')
    x=np.arange(len(names)); width=.36
    for offset,mode,color in [(-width/2,'baseline','#a2adbd'),(width/2,'tuned','#3177c6')]:
        values=[summary['engine'][name][mode] for name in names]
        axes[0].bar(x+offset,values,width,label=mode,color=color)
    for i,name in enumerate(names):
        axes[0].annotate(f"{summary['engine'][name]['speedup']:.2f}×",(i,max(summary['engine'][name]['baseline'],
            summary['engine'][name]['tuned'])+2),ha='center',fontsize=10)
    axes[0].set(xticks=x,xticklabels=names,ylabel='Engine latency (ms)',title='27B INT8: median decision latency')
    axes[0].legend(frameon=False);axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)

    base=load('qualified-baseline.json')['projections']
    table=json.loads((Path(__file__).resolve().parents[1]/'startlux_prism/kernels/w8a8_table_gfx950.json').read_text())
    for mode,color in [('baseline','#a2adbd'),('tuned','#3177c6')]:
        intensities,rates=[],[]
        for row in base:
            if row['m']<64: continue
            m,n,k,epi=row['m'],row['n'],row['k'],row['epi']
            key=f'{m},{n},{k},{epi}'
            ms=(table['measurements'][key]['selected_us'] if mode=='tuned' and key in table['measurements'] else row['us'])/1000
            ops=2*m*n*k
            minimum_bytes=n*k+m*k+m*n*(8 if epi else 2)+4*(m+n)
            intensities.append(ops/minimum_bytes);rates.append(ops/(ms*1e9))
        axes[1].scatter(intensities,rates,s=21,label=mode,color=color,alpha=.65)
    intensity=np.geomspace(2,3000,200)
    bw=ceilings['ceilings']['triton_copy']['tb_s']
    compute=ceilings['ceilings']['torch.int8']['tera_ops_s']
    axes[1].plot(intensity,np.minimum(bw*intensity,compute),color='#1d7358',label='Measured reference envelope')
    axes[1].plot(intensity,np.minimum(8*intensity,5033.2),ls='--',color='#333333',label='MI355X theoretical envelope')
    axes[1].set(xscale='log',yscale='log',xlabel='Useful operations / minimum tensor bytes',ylabel='Useful INT8 TOPS',
                title='Projection throughput; weights streamed beyond LLC')
    axes[1].legend(frameon=False,fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('AMD Instinct MI355X · exact INT8 tiles · prefix cache disabled',fontsize=13)
    fig.savefig(root/'mi355x-performance.svg')
    svg=root/'mi355x-performance.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(root/'mi355x-performance.png',dpi=160)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
