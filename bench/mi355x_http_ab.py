"""Measure baseline and tuned INT8 Docker servers with unique cache-disabled inference."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

from run_http import Client, make_body
import suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True)
    parser.add_argument('--gguf',required=True,help='basename inside the model directory')
    parser.add_argument('--calib',required=True)
    parser.add_argument('--image',default='startlux-prism:mi355x')
    parser.add_argument('--port',type=int,default=18112)
    parser.add_argument('--n',type=int,default=20)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    model, calib = Path(args.model_dir).resolve(), Path(args.calib).resolve()
    report, reference = dict(measurements=[]), {}
    container = 'startlux-prism-mi355x-perf-http'
    for mode in ('baseline','tuned'):
        command=['docker','run','--rm','--name',container,'--network','host','--ipc=host',
                 '--device=/dev/kfd','--device=/dev/dri','--group-add','video',
                 '--security-opt','seccomp=unconfined','-e','HIP_VISIBLE_DEVICES='+os.getenv('HIP_VISIBLE_DEVICES','0'),
                 '-e','STARTLUX_PRISM_MI355X_GEMM='+mode,'-v',str(model)+':/models:ro',
                 '-v',str(calib)+':/calib.pt:ro','-v','startlux-prism-mi355x-perf-cache:/root/.triton',
                 args.image,'--gguf','/models/'+args.gguf,'--model-dir','/models','--calib','/calib.pt',
                 '--port',str(args.port),'--max-tokens','4096','--graphs','--graph-max-tokens','1536',
                 '--prefix-cache','0','--mmproj','none','--think-mode','never','--max-think-budget','64',
                 '--think-seqs','1','--think-kv-rows','4096']
        with output.with_name('http-server-'+mode+'.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                base='http://127.0.0.1:'+str(args.port)
                deadline=time.monotonic()+600
                while True:
                    if process.poll() is not None: raise RuntimeError('server exited: see '+log.name)
                    try:
                        health=json.load(urllib.request.urlopen(base+'/health',timeout=2))
                        break
                    except OSError:
                        if time.monotonic()>deadline: raise TimeoutError('server startup')
                        time.sleep(.25)
                print('READY',mode,flush=True)
                report[mode+'_health']=health
                client=Client(base+'/v1/systemone')
                for name in ('yesno','review1','readme3','startlux3','longpage2'):
                    state,questions=suites.fixed()[name]
                    body=json.loads(make_body(state,questions,unique=False))
                    body['think_mode']='never'
                    body=json.dumps(body).encode()
                    for _ in range(3): _, answer=client.call(body)
                    if name not in reference: reference[name]=(answer['answers'],answer['usage'])
                    assert (answer['answers'],answer['usage'])==reference[name],(mode,name,'answers changed')
                    for repeat in range(3):
                        samples=[client.call(body)[0] for _ in range(args.n)]
                        row=dict(mode=mode,suite=name,repeat=repeat,median_ms=statistics.median(samples),
                                 samples_ms=samples,exact_answers=True)
                        report['measurements'].append(row)
                        output.write_text(json.dumps(report,indent=2)+'\n')
                        print(mode,name,repeat,round(row['median_ms'],3),'ms',flush=True)
            finally:
                subprocess.run(['docker','stop','--timeout','5',container],stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                process.wait(timeout=20)


if __name__=='__main__':
    main()
