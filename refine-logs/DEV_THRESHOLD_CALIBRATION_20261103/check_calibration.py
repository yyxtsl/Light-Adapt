"""Small independent fixtures; no dataset/model access or GPU calls."""
import json
from fractions import Fraction
import numpy as np
from calibration_core import scan, fixed_grid, margins, make_gold_mask, prediction_records


def run():
    checks=[]
    rng=np.random.default_rng(751103)
    for size in [1,5,31]:
        logits=rng.integers(-4,5,size=(size,4)).astype(np.float32)
        y=rng.integers(0,2,size=(size,3)).astype(bool)
        grid=np.arange(-12,13,dtype=float)/2
        result=scan(logits,y,grid)
        expected=[]
        for d,row in zip(grid,result['rows']):
            p=logits[:,1:].astype(float)>logits[:,:1].astype(float)+d
            tp=int((p&y).sum());pred=int(p.sum());gold=int(y.sum())
            assert (row['TP'],row['FP'],row['FN'])==(tp,pred-tp,gold-tp)
            expected.append((Fraction(2*tp,pred+gold) if pred+gold else Fraction(0),-abs(d),d))
        assert result['selected']['delta']==grid[max(range(len(grid)),key=lambda i:expected[i])]
        checks.append('independent brute-force scan '+str(size))
    assert len(fixed_grid())==4001 and fixed_grid()[2000]==0 and fixed_grid()[0]==-100 and fixed_grid()[-1]==100
    checks.append('fixed symmetric 4001-point grid includes zero')
    tie=scan(np.array([[0,1]],dtype=np.float32),np.array([[True]]),np.array([-.5,.5]))
    assert tie['selected']['delta']==.5
    checks.append('tie: smallest absolute delta, then larger delta')
    zero=scan(np.zeros((2,3),dtype=np.float32),np.zeros((2,2),bool),np.array([-1.,0.,1.]))
    assert zero['selected']['delta']==0 and zero['selected']['F1']==0
    checks.append('zero-gold/zero-prediction cases and strict tie')
    logits=np.array([[1,1,2],[0,-1,0]],dtype=np.float32)
    assert np.array_equal(margins(logits)>0,logits[:,1:]>logits[:,:1])
    checks.append('delta-zero exactly matches native TH decision')
    assert np.array_equal(logits[:,1:]>logits[:,:1],(logits/2)[:,1:]>(logits/2)[:,:1])
    checks.append('uniform positive temperature leaves decisions unchanged')
    docs=[{'title':'fixture','vertexSet':[[],[]],'labels':[{'h':0,'t':1,'r':'P1'}]}]
    order=[{'title':'fixture','hts':[[0,1],[1,0]]}]
    mask=make_gold_mask(order,docs,{'TH':0,'P1':1,'P2':2})
    assert mask.tolist()==[[True,False],[False,False]]
    assert prediction_records(np.array([[0,1,-1],[0,0,2]],dtype=np.float32),0,order,{1:'P1',2:'P2'})==[
        {'title':'fixture','h_idx':0,'t_idx':1,'r':'P1'},
        {'title':'fixture','h_idx':1,'t_idx':0,'r':'P2'}]
    checks.append('gold and prediction mapping to exact native pair order')
    for value in [np.array([[0,np.nan]]),np.array([[0,np.inf]])]:
        try:margins(value)
        except ValueError:pass
        else:raise AssertionError('Accepted nonfinite logits')
    checks.append('reject nonfinite logits')
    try:make_gold_mask([{'title':'fixture','hts':[[0,1]]}],docs,{'TH':0,'P1':1,'P2':2})
    except ValueError:pass
    else:raise AssertionError('Accepted candidate filtering')
    checks.append('reject candidate filtering')
    print(json.dumps({'passed':True,'checks':checks,'synthetic_only':True,'GPU_forwards':0,'real_checkpoint_opens':0},indent=2))
if __name__=='__main__':run()
