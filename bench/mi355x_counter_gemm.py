"""Four different weight matrices for counter collection; select dispatch iteration 4."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from startlux_prism.kernels import w8a8 as W


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('baseline','tuned'),default='baseline')
    args=parser.parse_args()
    os.environ['STARTLUX_PRISM_MI355X_GEMM']=args.mode
    m,n,k=192,34816,5120
    x=torch.randint(-127,128,(m,k),device='cuda',dtype=torch.int8)
    weights=[torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8) for _ in range(4)]
    sa=torch.full((m,),0.01,device='cuda')
    sw=torch.full((n,),0.001,device='cuda')
    out=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
    live=torch.tensor([m],device='cuda',dtype=torch.int32)
    for w in weights:
        W.gemm(x,sa,w,sw,out,live,k)
        torch.cuda.synchronize()
    print(args.mode,W.pick(m,n,k,0),flush=True)


if __name__=='__main__':
    main()
