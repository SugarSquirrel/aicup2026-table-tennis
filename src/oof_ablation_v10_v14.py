"""OOF ablation of the post-v9 components for the paper (post-competition experiment).
Fills Table III rows (e)-(h): multi-seed GRUs, noise filter, old-test samples in training (v14),
and old-test samples for the action models only (v14g), under the same stratified 5-fold CV as v7/v8.
Requires oof_v7_actionaug.npz and oof_v8_t2v_film.npz (run gen_submission_v7_actionaug.py / v8 first).
Shared helpers below are copied verbatim from gen_submission_v14_old_in_training.py.
"""
import time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb, torch, torch.nn as nn
from tabpfn import TabPFNClassifier
from tabpfn_extensions.many_class import ManyClassClassifier
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED); DEV="cuda"
tr=pd.read_csv('../data/train.csv'); te=pd.read_csv('../data/test_new.csv')
old=pd.read_csv('../data/Reference_Only_Old_Test_Data/test.csv')
STROKE=["strikeId","handId","strengthId","spinId","pointId","actionId","positionId"]; REC=["scoreSelf","scoreOther"]+STROKE
ACLS=np.arange(19); PCLS=np.arange(10)
def out(*a): print(*a)
def feats(strokes,sex,L):
    last=strokes[L-1]; f={"sex":sex,"obs_len":L,"obs_parity":L%2,"next_is_server":(L+1)%2,"score_self":last["scoreSelf"],"score_other":last["scoreOther"],"score_diff":last["scoreSelf"]-last["scoreOther"],"score_sum":last["scoreSelf"]+last["scoreOther"]}
    for off in range(1,4):
        tag="last" if off==1 else f"lag{off}"
        if L>=off:
            r=strokes[L-off]
            for c in STROKE: f[f"{tag}_{c}"]=r[c]
        else:
            for c in STROKE: f[f"{tag}_{c}"]=-1
    f["mean_spin"]=float(np.mean([strokes[i]["spinId"] for i in range(L)])); f["mean_strength"]=float(np.mean([strokes[i]["strengthId"] for i in range(L)]))
    f["nuniq_point"]=len({strokes[i]["pointId"] for i in range(L)}); f["nuniq_action"]=len({strokes[i]["actionId"] for i in range(L)})
    f["_la"]=last["actionId"]; f["_lp"]=last["pointId"]
    return f
def build(df,mode,tld,seed=SEED,test_mode=False,exclude_uids=None):
    rng=np.random.default_rng(seed)
    if mode=="sampled": Ls=np.array(sorted(tld)); Ps=np.array([tld[l] for l in Ls],float); Ps/=Ps.sum()
    rows,yA,yP,yR,nh,lh,uid,g=[],[],[],[],[],[],[],[]
    for rid,grp in df.groupby("rally_uid",sort=False):
        if (exclude_uids is not None) and (int(rid) in exclude_uids): continue   # v12: skip noisy rallies
        grp=grp.sort_values("strikeNumber"); T=len(grp); st=grp[REC].to_dict("records"); gp=grp.gamePlayerId.to_numpy(); go=grp.gamePlayerOtherId.to_numpy()
        if test_mode: Ll=[T]
        else:
            if T<2: continue
            na=grp.actionId.to_numpy(); npt=grp.pointId.to_numpy(); sgp=int(grp.serverGetPoint.iloc[0]); mt=int(grp.match.iloc[0])
            Ll=range(1,T) if mode=="all" else ([1] if len(Ls[Ls<=T-1])==0 else [int(rng.choice(Ls[Ls<=T-1],p=(Ps[Ls<=T-1]/Ps[Ls<=T-1].sum())))])
        for L in Ll:
            rows.append(feats(st,int(grp.sex.iloc[0]),L)); nh.append(int(go[L-1])); lh.append(int(gp[L-1]))
            if test_mode: uid.append(int(rid))
            else: yA.append(int(na[L])); yP.append(int(npt[L])); yR.append(sgp); g.append(mt)
    X=pd.DataFrame(rows)
    if test_mode: return X,np.array(nh),np.array(lh),np.array(uid)
    return X,np.array(yA),np.array(yP),np.array(yR),np.array(nh),np.array(lh),np.array(g)
def fit_trans(keys,yA,yP,alpha=1.0):
    def cd(ka,y,nc):
        d={}
        for k,yy in zip(ka,y): d.setdefault(k,np.zeros(nc))[yy]+=1
        gp=np.bincount(y,minlength=nc)+alpha; gp/=gp.sum(); return {k:(v+alpha)/(v.sum()+alpha*nc) for k,v in d.items()},gp
    lalp=list(zip(keys["_la"],keys["_lp"])); return dict(aJ=cd(lalp,yA,19),pJ=cd(lalp,yP,10))
def apply_trans(keys,T):
    out={}; lalp=list(zip(keys["_la"],keys["_lp"]))
    d,gp=T["aJ"]; M=np.array([d.get(k,gp) for k in lalp])
    for j in range(19): out[f"tA_{j}"]=M[:,j]
    d,gp=T["pJ"]; M=np.array([d.get(k,gp) for k in lalp])
    for j in range(10): out[f"tP_{j}"]=M[:,j]
    return pd.DataFrame(out)
def player_dists(nh,yA,yP,alpha=10):
    dA={};dP={}
    for h,a in zip(nh,yA): dA.setdefault(h,np.zeros(19))[a]+=1
    for h,p in zip(nh,yP): dP.setdefault(h,np.zeros(10))[p]+=1
    gA=np.bincount(yA,minlength=19)+1.;gA/=gA.sum();gP=np.bincount(yP,minlength=10)+1.;gP/=gP.sum()
    return ({h:(v+alpha*gA)/(v.sum()+alpha) for h,v in dA.items()},gA,{h:(v+alpha*gP)/(v.sum()+alpha) for h,v in dP.items()},gP)
def fit_clusters(df_sub,k=6):
    vecs={}
    for pl,g in df_sub.groupby('gamePlayerId'):
        a=np.bincount(g.actionId,minlength=19).astype(float);a/=max(a.sum(),1)
        p=np.bincount(g.pointId,minlength=10).astype(float);p/=max(p.sum(),1)
        vecs[pl]=np.concatenate([a,p])
    pls=list(vecs);km=KMeans(k,n_init=10,random_state=SEED).fit(np.array([vecs[p] for p in pls]))
    return {p:int(c) for p,c in zip(pls,km.labels_)}
def fit_matchup(nh,lh,yA,yP,cl):
    cA={};cP={}
    for h,o,a,p in zip(nh,[cl.get(x,-1) for x in lh],yA,yP):
        cA.setdefault((h,o),np.zeros(19))[a]+=1; cP.setdefault((h,o),np.zeros(10))[p]+=1
    return cA,cP
def matchup_feat(nh,lh,cl,cA,cP,dMa,gA,dMp,gP,idx,a=8):
    ocs=[cl.get(x,-1) for x in lh]
    MA=np.array([(cA.get((h,o),np.zeros(19))+a*dMa.get(h,gA))/(cA.get((h,o),np.zeros(19)).sum()+a) for h,o in zip(nh,ocs)])
    MP=np.array([(cP.get((h,o),np.zeros(10))+a*dMp.get(h,gP))/(cP.get((h,o),np.zeros(10)).sum()+a) for h,o in zip(nh,ocs)])
    return pd.DataFrame({**{f'mcA{j}':MA[:,j] for j in range(19)},**{f'mcP{j}':MP[:,j] for j in range(10)}},index=idx)
def player_feat(nh,dA,gA,dP,gP,idx):
    MA=np.array([dA.get(h,gA) for h in nh]); MP=np.array([dP.get(h,gP) for h in nh])
    return pd.DataFrame({**{f'phA{j}':MA[:,j] for j in range(19)},**{f'phP{j}':MP[:,j] for j in range(10)}},index=idx)
def lgbc(bal=True): return lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=63,subsample=0.8,colsample_bytree=0.8,class_weight=("balanced" if bal else None),random_state=SEED,n_jobs=-1,verbose=-1)
def align(p,c,full):
    o=np.zeros((p.shape[0],len(full))); idx={cc:i for i,cc in enumerate(c)}
    for j,cc in enumerate(full):
        if cc in idx: o[:,j]=p[:,idx[cc]]
    return o
CAT=["actionId","pointId","spinId","strengthId","handId","positionId","strikeId"]
VOCAB={c:int(tr[c].max())+2 for c in CAT}; VOCAB['role']=3; VOCAB['sex']=int(tr.sex.max())+2; NCAT=len(CAT)+2; MAXLEN=30
def rseq(g):
    g=g.sort_values("strikeNumber")
    cat=np.stack([g[c].to_numpy()+1 for c in CAT]+[(g.strikeNumber.to_numpy()%2)+1,np.full(len(g),int(g.sex.iloc[0])+1)],axis=1)
    num=np.stack([g.scoreSelf.to_numpy()/10.,g.scoreOther.to_numpy()/10.,g.strikeNumber.to_numpy()/15.],axis=1)
    sgp=int(g.serverGetPoint.iloc[0]) if "serverGetPoint" in g.columns else 0
    return cat.astype(np.int64),num.astype(np.float32),g.actionId.to_numpy(),g.pointId.to_numpy(),sgp
def build_seq(df,mode,tld,seed=SEED,test_mode=False,exclude_uids=None):
    rng=np.random.default_rng(seed)
    if mode=="sampled": Ls=np.array(sorted(tld));Ps=np.array([tld[l] for l in Ls],float);Ps/=Ps.sum()
    C=[];Nu=[];Ln=[];yA=[];yP=[];yR=[]
    for rid,grp in df.groupby("rally_uid",sort=False):
        if (exclude_uids is not None) and (int(rid) in exclude_uids): continue   # v12: skip noisy rallies
        cat,num,na,npt,sgp=rseq(grp);T=len(na)
        if test_mode: Ll=[T]
        else:
            if T<2: continue
            Ll=range(1,T) if mode=="all" else ([1] if len(Ls[Ls<=T-1])==0 else [int(rng.choice(Ls[Ls<=T-1],p=(Ps[Ls<=T-1]/Ps[Ls<=T-1].sum())))])
        for L in Ll:
            l=min(L,MAXLEN); pc=np.zeros((MAXLEN,NCAT),np.int64); pn=np.zeros((MAXLEN,3),np.float32)
            pc[:l]=cat[L-l:L]; pn[:l]=num[L-l:L]; C.append(pc); Nu.append(pn); Ln.append(l)
            if not test_mode: yA.append(int(na[L])); yP.append(int(npt[L])); yR.append(sgp)
    if test_mode: return np.stack(C),np.stack(Nu),np.array(Ln)
    return np.stack(C),np.stack(Nu),np.array(Ln),np.array(yA),np.array(yP),np.array(yR)

class GRUNetV7(nn.Module):
    def __init__(s):
        super().__init__()
        s.embs=nn.ModuleList([nn.Embedding(VOCAB[c],8,padding_idx=0) for c in CAT]+[nn.Embedding(VOCAB['role'],4,padding_idx=0),nn.Embedding(VOCAB['sex'],4,padding_idx=0)])
        s.num=nn.Linear(3,16); s.gru=nn.GRU(8*len(CAT)+4+4+16,64,batch_first=True); s.drop=nn.Dropout(0.2)
        s.ha=nn.Linear(64,19); s.hp=nn.Linear(64,10); s.hs=nn.Linear(64,1)
    def forward(s,cat,num,ln):
        e=torch.cat([s.embs[i](cat[:,:,i]) for i in range(NCAT)]+[torch.relu(s.num(num))],-1)
        pk=nn.utils.rnn.pack_padded_sequence(e,ln.cpu(),batch_first=True,enforce_sorted=False)
        _,h=s.gru(pk); h=s.drop(h[-1])
        return s.ha(h),s.hp(h),s.hs(h).squeeze(1)
class Time2Vec(nn.Module):
    def __init__(s,in_dim,k=7):
        super().__init__()
        s.w_lin=nn.Parameter(torch.randn(in_dim)*0.1); s.b_lin=nn.Parameter(torch.zeros(in_dim))
        s.W=nn.Parameter(torch.randn(in_dim,k)*0.5); s.B=nn.Parameter(torch.randn(in_dim,k)*0.5)
    def forward(s,x):
        lin=(x*s.w_lin+s.b_lin).unsqueeze(-1); per=torch.sin(x.unsqueeze(-1)*s.W+s.B)
        return torch.cat([lin,per],-1).flatten(-2)
class FiLM(nn.Module):
    def __init__(s,ctx_dim,feat_dim):
        super().__init__(); s.proj=nn.Linear(ctx_dim,feat_dim*2)
    def forward(s,feat,ctx):
        gb=s.proj(ctx); g,b=gb.chunk(2,-1)
        return feat*(1.0+torch.tanh(g))+b
class GRUNetV8(nn.Module):
    def __init__(s):
        super().__init__()
        s.embs=nn.ModuleList([nn.Embedding(VOCAB[c],8,padding_idx=0) for c in CAT]+[nn.Embedding(VOCAB['role'],4,padding_idx=0),nn.Embedding(VOCAB['sex'],4,padding_idx=0)])
        s.t2v=Time2Vec(3,k=7); s.gru=nn.GRU(8*len(CAT)+4+4+24,64,batch_first=True); s.drop=nn.Dropout(0.2)
        s.film=FiLM(ctx_dim=4+24+4, feat_dim=64)
        s.ha=nn.Linear(64,19); s.hp=nn.Linear(64,10); s.hs=nn.Linear(64,1)
    def forward(s,cat,num,ln):
        emb_list=[s.embs[i](cat[:,:,i]) for i in range(NCAT)]; t2v=s.t2v(num)
        e=torch.cat(emb_list+[t2v],-1)
        pk=nn.utils.rnn.pack_padded_sequence(e,ln.cpu(),batch_first=True,enforce_sorted=False)
        _,h=s.gru(pk); h=s.drop(h[-1])
        sex_emb=s.embs[NCAT-1](cat[:,0,NCAT-1]); role_emb=s.embs[NCAT-2](cat[:,:,NCAT-2]).mean(1)
        mask=(torch.arange(t2v.size(1),device=t2v.device).unsqueeze(0)<ln.unsqueeze(1)).unsqueeze(-1).float()
        t2v_pool=(t2v*mask).sum(1)/mask.sum(1).clamp_min(1.0)
        ctx=torch.cat([sex_emb,t2v_pool,role_emb],-1)
        h=s.film(h,ctx)
        return s.ha(h),s.hp(h),s.hs(h).squeeze(1)
def _cw(y,n): c=np.bincount(y,minlength=n)+1; w=1./c; return torch.tensor(w*n/w.sum(),dtype=torch.float32,device=DEV)
def gru_train(Xc,Xn,Xl,yA,yP,yR,ep=12,Cls=GRUNetV7):
    torch.manual_seed(SEED); m=Cls().to(DEV); opt=torch.optim.Adam(m.parameters(),1e-3)
    cea=nn.CrossEntropyLoss(weight=_cw(yA,19));cep=nn.CrossEntropyLoss(weight=_cw(yP,10));bce=nn.BCEWithLogitsLoss()
    Cc=torch.tensor(Xc,device=DEV); Nn=torch.tensor(Xn,device=DEV); Ll=torch.tensor(Xl,device=DEV)
    Ta=torch.tensor(yA,device=DEV); Tp=torch.tensor(yP,device=DEV); Tr=torch.tensor(yR.astype('float32'),device=DEV)
    ii=np.arange(len(yA))
    for e in range(ep):
        m.train(); np.random.shuffle(ii)
        for i in range(0,len(ii),256):
            b=ii[i:i+256]; opt.zero_grad(); la,lp,lr=m(Cc[b],Nn[b],Ll[b])
            (0.4*cea(la,Ta[b])+0.4*cep(lp,Tp[b])+0.2*bce(lr,Tr[b])).backward(); opt.step()
    return m
def gru_pred(m,Xc,Xn,Xl):
    m.eval(); A=[]; P=[]; R=[]
    with torch.no_grad():
        for i in range(0,len(Xl),512):
            la,lp,lr=m(torch.tensor(Xc[i:i+512],device=DEV),torch.tensor(Xn[i:i+512],device=DEV),torch.tensor(Xl[i:i+512],device=DEV))
            A.append(torch.softmax(la,1).cpu().numpy()); P.append(torch.softmax(lp,1).cpu().numpy()); R.append(torch.sigmoid(lr).cpu().numpy())
    return np.vstack(A),np.vstack(P),np.concatenate(R)

# ===== Load v7+v8 OOF caches, blend GRU, search weights =====
v7=np.load('oof_v7_actionaug.npz'); v8=np.load('oof_v8_t2v_film.npz')
YA=v7['YA']; YP=v7['YP']; YR=v7['YR']
SEEN=v7['SEEN'].astype(bool); RES=v7['RESCUED'].astype(bool); COLD=v7['COLD'].astype(bool)
LAc=v7['LAc']; LPc=v7['LPc']; LRc=v7['LRc']; TAc=v7['TAc']; TPc=v7['TPc']; TRc=v7['TRc']
GRU_BLEND=0.5  # 0.5 wins on OOF blend diagnostic
GAc=GRU_BLEND*v7['GAc']+(1-GRU_BLEND)*v8['GAc']
GPc=GRU_BLEND*v7['GPc']+(1-GRU_BLEND)*v8['GPc']
GRc=GRU_BLEND*v7['GRc']+(1-GRU_BLEND)*v8['GRc']
prA=np.array([(YA==c).mean() for c in ACLS]); prP=np.array([(YP==c).mean() for c in PCLS])
BETA_GRID=np.linspace(0,2.5,21); WEIGHT_STEP=0.05
def bb(p,y,cls,pr):
    fb=-1; b0=0
    for b in BETA_GRID:
        ff=f1_score(y,cls[(p/np.clip(pr,1e-9,None)**b).argmax(1)],average="macro")
        if ff>fb: fb=ff; b0=b
    return b0,fb
def f1m(p,y,cls,pr,b,m): return f1_score(y[m],cls[(p[m]/np.clip(pr,1e-9,None)**b).argmax(1)],average="macro")
def s3(L,T,G,y,cls,pr):
    best=(-1,None,0,None)
    for wl in np.arange(0,1.0001,WEIGHT_STEP):
        for wt in np.arange(0,1.0001-wl,WEIGHT_STEP):
            wg=round(1-wl-wt,4)
            if wg<-1e-9 or wg>1.0001: continue
            bl=wl*L+wt*T+wg*G; b,_=bb(bl,y,cls,pr)
            cvb=0.94*f1m(bl,y,cls,pr,b,SEEN)+0.06*f1m(bl,y,cls,pr,b,~SEEN)
            cva=f1_score(y,cls[(bl/np.clip(pr,1e-9,None)**b).argmax(1)],average="macro")
            if cvb>best[0]: best=(cvb,(round(wl,3),round(wt,3),round(wg,3)),b,cva)
    return best
fa_b,WA,BA,fa_a=s3(LAc,TAc,GAc,YA,ACLS,prA); fp_b,WP,BP,fp_a=s3(LPc,TPc,GPc,YP,PCLS,prP)
best_auc=-1; WR=None
for wl in np.arange(0,1.0001,WEIGHT_STEP):
    for wt in np.arange(0,1.0001-wl,WEIGHT_STEP):
        wg=round(1-wl-wt,4)
        if wg<-1e-9 or wg>1.0001: continue
        a=roc_auc_score(YR,wl*LRc+wt*TRc+wg*GRc)
        if a>best_auc: best_auc=a; WR=(round(wl,3),round(wt,3),round(wg,3))
out(f"v14 (filter+old-train+blend α={GRU_BLEND}) CV-B: action F1={fa_b:.4f} w={WA} β={BA:.3f}  | CV-A={fa_a:.4f}")
out(f"                                   point  F1={fp_b:.4f} w={WP} β={BP:.3f}  | CV-A={fp_a:.4f}")
out(f"                                   server AUC={best_auc:.4f} w={WR}")
out(f"                                   => CV-B Overall = {0.4*fa_b+0.4*fp_b+0.2*best_auc:.4f}")
# per-stratum
def f1_strat(L,T,G,y,cls,pr,W,B,mask):
    if mask.sum()<5: return float('nan')
    bl=W[0]*L+W[1]*T+W[2]*G
    return f1_score(y[mask],cls[(bl[mask]/np.clip(pr,1e-9,None)**B).argmax(1)],average="macro")
out("\n[per-stratum]")
for name,m in [("seen",SEEN),("rescued",RES),("cold",COLD)]:
    fa=f1_strat(LAc,TAc,GAc,YA,ACLS,prA,WA,BA,m); fp=f1_strat(LPc,TPc,GPc,YP,PCLS,prP,WP,BP,m)
    out(f"  {name:7s} (n={m.sum():>5d}):  action={fa:.4f}  point={fp:.4f}")

# ===== Train on FULL + predict test (v7 task gating + blended GRU) =====
tld=te.groupby('rally_uid').size().value_counts().to_dict()

# v12 ★ identify noisy training rallies via v10-style OOF ensemble (action prob<0.05 to true label)
NOISE_THRESHOLD=0.05
# Use already-loaded v7/v8 OOF caches (loaded at top of script)
_PA_oof=0.35*v7['LAc']+0.25*v7['TAc']+0.40*(0.5*v7['GAc']+0.5*v8['GAc'])
_prob_true=_PA_oof[np.arange(len(v7['YA'])),v7['YA']]
# Reconstruct sampled rally_uid in v7-OOF concat order
_Xs_tmp,_eA_tmp,_eP_tmp,_eR_tmp,_nhs_tmp,_lhs_tmp,_gs_tmp=build(tr,"sampled",tld)
_rng2=np.random.default_rng(SEED)
_uid_sampled=[]
_Ls2=np.array(sorted(tld));_Ps2=np.array([tld[l] for l in _Ls2],float);_Ps2/=_Ps2.sum()
for rid,grp in tr.groupby('rally_uid',sort=False):
    grp=grp.sort_values("strikeNumber"); T=len(grp)
    if T<2: continue
    cand=_Ls2[_Ls2<=T-1]
    if len(cand)==0: _L=1
    else: pr=_Ps2[_Ls2<=T-1]; _L=int(_rng2.choice(cand,p=pr/pr.sum()))
    _uid_sampled.append(int(rid))
_uid_sampled=np.array(_uid_sampled)
# fold mapping (must match v7 OOF construction)
_M=np.array(sorted(set(_gs_tmp)))
_gkf=GroupKFold(5); _fo={}
# Need to include all matches that appear in ga or gs. Use gs (sampled match per rally).
# Actually v7 uses M=sorted(set(ga)|set(gs)). Since gs⊆ga's rally matches usually:
_Xa_tmp,_yA_tmp,_yP_tmp,_yR_tmp,_nha_tmp,_lha_tmp,_ga_tmp=build(tr,"all",tld)
_M=np.array(sorted(set(_ga_tmp)|set(_gs_tmp)))
for _f,(_,_vi) in enumerate(_gkf.split(_M,groups=_M)):
    for _m in _M[_vi]: _fo[_m]=_f
_sf=np.array([_fo[m] for m in _gs_tmp])
# OOF concat order = sorted by fold
_oof_to_sampled=np.concatenate([np.where(_sf==f)[0] for f in range(5)])
_uid_oof=_uid_sampled[_oof_to_sampled]
# Sanity: v7['YA'] order should match
assert (_eA_tmp[_oof_to_sampled]==v7['YA']).all(), "OOF label alignment mismatch"
EXCLUDE_UIDS=set(_uid_oof[_prob_true<NOISE_THRESHOLD].tolist())
out(f"v12 noise filter: {len(EXCLUDE_UIDS)}/{len(_uid_oof)} rallies excluded (prob_true_action < {NOISE_THRESHOLD})")
del _PA_oof,_prob_true,_uid_sampled,_uid_oof,_oof_to_sampled,_sf,_M,_Xs_tmp,_Xa_tmp

# ===== OOF ablation of the post-v9 components (paper Table III rows e-h) =====
# Same 5 folds and sampled eval prefixes as the v7/v8 OOF caches. Per fold:
#   (e) v9 + multi-seed GRUs (3 seeds x 2 archs); LGBM/TabPFN = cached v7 OOF
#   (f) (e) + noise filter on the training folds (eval rallies are never filtered)
#   (g) (f) + old-test samples in training for all tasks            (= v14 in CV)
#   (h) old-test samples for the action models only: action from (g), point/server from (f)  (= v14g in CV)
# Ensemble weights/betas stay fixed to the v9 search above, exactly as v14 used them.
import os
SEEDS=[42,1,2]
def multiseed_train_predict(Cls,Ca,Na,La_,gyA,gyP,gyR,Cte,Nte,Lte,seeds):
    gA_acc,gP_acc,gR_acc=None,None,None
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m=Cls().to(DEV); opt=torch.optim.Adam(m.parameters(),1e-3)
        cea=nn.CrossEntropyLoss(weight=_cw(gyA,19)); cep=nn.CrossEntropyLoss(weight=_cw(gyP,10)); bce=nn.BCEWithLogitsLoss()
        Cc=torch.tensor(Ca,device=DEV); Nn=torch.tensor(Na,device=DEV); Ll=torch.tensor(La_,device=DEV)
        Ta=torch.tensor(gyA,device=DEV); Tp=torch.tensor(gyP,device=DEV); Tr=torch.tensor(gyR.astype('float32'),device=DEV)
        ii=np.arange(len(gyA))
        for e in range(12):
            m.train(); np.random.shuffle(ii)
            for i in range(0,len(ii),256):
                b=ii[i:i+256]; opt.zero_grad(); la,lp,lr=m(Cc[b],Nn[b],Ll[b])
                (0.4*cea(la,Ta[b])+0.4*cep(lp,Tp[b])+0.2*bce(lr,Tr[b])).backward(); opt.step()
        a,p,r=gru_pred(m,Cte,Nte,Lte)
        gA_acc=a if gA_acc is None else gA_acc+a
        gP_acc=p if gP_acc is None else gP_acc+p
        gR_acc=r if gR_acc is None else gR_acc+r
    n=len(seeds); return gA_acc/n, gP_acc/n, gR_acc/n
def blend_gru(Ca,Na,La_,gyA,gyP,gyR,Ce,Ne,Le):
    a7,p7,r7=multiseed_train_predict(GRUNetV7,Ca,Na,La_,gyA,gyP,gyR,Ce,Ne,Le,SEEDS)
    a8,p8,r8=multiseed_train_predict(GRUNetV8,Ca,Na,La_,gyA,gyP,gyR,Ce,Ne,Le,SEEDS)
    return GRU_BLEND*a7+(1-GRU_BLEND)*a8, GRU_BLEND*p7+(1-GRU_BLEND)*p8, GRU_BLEND*r7+(1-GRU_BLEND)*r8

# rally_uid of every sample, in build() order, so the noise filter can mask training samples
def sample_uids(df,mode):
    u=[]
    for rid,grp in df.groupby("rally_uid",sort=False):
        T=len(grp)
        if T<2: continue
        u.extend([int(rid)]*((T-1) if mode=="all" else 1))
    return np.array(u)
Xa,yA,yP,yR,nha,lha,ga=build(tr,"all",tld); Xs,eA,eP,eR,nhs,lhs,gs=build(tr,"sampled",tld)
Xao,yAo,yPo,yRo,nhao,lhao,gao=build(old,"all",tld); Xao_smp,eAo,ePo,eRo,nhao_s,lhao_s,gao_s=build(old,"sampled",tld)
ua=sample_uids(tr,"all"); us=sample_uids(tr,"sampled"); assert len(ua)==len(yA) and len(us)==len(eA)
keepA=~np.isin(ua,list(EXCLUDE_UIDS)); keepS=~np.isin(us,list(EXCLUDE_UIDS))
KEY=['_la','_lp']; BASE=[c for c in Xa.columns if c not in KEY]
M=np.array(sorted(set(ga)|set(gs))); gkf=GroupKFold(5); fo={}
for f,(_,vi) in enumerate(gkf.split(M,groups=M)):
    for m in M[vi]: fo[m]=f
af=np.array([fo[m] for m in ga]); sf=np.array([fo[m] for m in gs]); trfold=tr.match.map(fo)
assert set(old.match).isdisjoint(set(tr.match)) and set(old.rally_uid).isdisjoint(set(tr.rally_uid))
Call,Nall,Lall,gyAall,gyPall,gyRall=build_seq(tr,"sampled",tld)
assert (gyAall==eA).all() and (gyPall==eP).all() and (Lall==Xs["obs_len"].to_numpy()).all()
Ca_o,Na_o,La_o,gyA_o,gyP_o,gyR_o=build_seq(old,"all",tld)
# eval order of the OOF cache = concatenation of the sampled-eval slices fold by fold
assert np.array_equal(np.concatenate([eA[sf==f] for f in range(5)]),YA), "fold/eval order differs from OOF cache"
out(f"noise filter keeps {keepA.sum()}/{len(keepA)} all-prefix and {keepS.sum()}/{len(keepS)} sampled training samples")

def fold_makers(f,mask):
    """Fold-safe stats on the fold's training samples selected by mask (aug = +old for action, train-only for point/server)."""
    m=(af!=f)&mask; trows=tr[trfold!=f]
    cat_nh=np.concatenate([nha[m],nhao]); cat_lh=np.concatenate([lha[m],lhao]); cat_yA=np.concatenate([yA[m],yAo]); cat_yP=np.concatenate([yP[m],yPo])
    T_a=fit_trans({"_la":np.concatenate([Xa.loc[m,"_la"].to_numpy(),Xao["_la"].to_numpy()]),"_lp":np.concatenate([Xa.loc[m,"_lp"].to_numpy(),Xao["_lp"].to_numpy()])},cat_yA,cat_yP)
    dMa_a,gA_a,dMp_a,gP_a=player_dists(cat_nh,cat_yA,cat_yP); cl_a=fit_clusters(pd.concat([trows,old],ignore_index=True)); cA_a,cP_a=fit_matchup(cat_nh,cat_lh,cat_yA,cat_yP,cl_a)
    T_t=fit_trans({"_la":Xa.loc[m,"_la"].to_numpy(),"_lp":Xa.loc[m,"_lp"].to_numpy()},yA[m],yP[m])
    dMa_t,gA_t,dMp_t,gP_t=player_dists(nha[m],yA[m],yP[m]); cl_t=fit_clusters(trows); cA_t,cP_t=fit_matchup(nha[m],lha[m],yA[m],yP[m],cl_t)
    def tr_(Xb,T): Ft=apply_trans({k:Xb[k].to_numpy() for k in KEY},T); Ft.index=Xb.index; return Ft
    return dict(
        A=lambda Xb,nh,lh: pd.concat([Xb[BASE],tr_(Xb,T_a),player_feat(nh,dMa_a,gA_a,dMp_a,gP_a,Xb.index),matchup_feat(nh,lh,cl_a,cA_a,cP_a,dMa_a,gA_a,dMp_a,gP_a,Xb.index)],axis=1),
        P=lambda Xb,nh,lh: pd.concat([Xb[BASE],tr_(Xb,T_t),player_feat(nh,dMa_t,gA_t,dMp_t,gP_t,Xb.index),matchup_feat(nh,lh,cl_t,cA_t,cP_t,dMa_t,gA_t,dMp_t,gP_t,Xb.index)],axis=1),
        Aps=lambda Xb,nh: pd.concat([Xb[BASE],tr_(Xb,T_a),player_feat(nh,dMa_a,gA_a,dMp_a,gP_a,Xb.index)],axis=1),
        Pps=lambda Xb,nh: pd.concat([Xb[BASE],tr_(Xb,T_t),player_feat(nh,dMa_t,gA_t,dMp_t,gP_t,Xb.index)],axis=1),
        S=lambda Xb: pd.concat([Xb[BASE],tr_(Xb,T_t)],axis=1))
def tab_lgbm_tabpfn(f,mk,mA_,mS_,add_old):
    """LGBM + TabPFN for one fold; training samples selected by masks, optionally + old samples. Returns eval probs."""
    evm=sf==f; trA=(af!=f)&mA_; trS=(sf!=f)&mS_
    XtA,XtP,XtS=mk['A'](Xa.loc[trA],nha[trA],lha[trA]),mk['P'](Xa.loc[trA],nha[trA],lha[trA]),mk['S'](Xa.loc[trA])
    ytA,ytP,ytR=yA[trA],yP[trA],yR[trA]
    XpA,XpP,XpS=mk['Aps'](Xs.loc[trS],nhs[trS]),mk['Pps'](Xs.loc[trS],nhs[trS]),mk['S'](Xs.loc[trS])
    epA,epP,epR=eA[trS],eP[trS],eR[trS]
    if add_old:
        XtA=pd.concat([XtA,mk['A'](Xao,nhao,lhao)],ignore_index=True); XtP=pd.concat([XtP,mk['P'](Xao,nhao,lhao)],ignore_index=True); XtS=pd.concat([XtS,mk['S'](Xao)],ignore_index=True)
        ytA,ytP,ytR=np.concatenate([ytA,yAo]),np.concatenate([ytP,yPo]),np.concatenate([ytR,yRo])
        XpA=pd.concat([XpA,mk['Aps'](Xao_smp,nhao_s)],ignore_index=True); XpP=pd.concat([XpP,mk['Pps'](Xao_smp,nhao_s)],ignore_index=True); XpS=pd.concat([XpS,mk['S'](Xao_smp)],ignore_index=True)
        epA,epP,epR=np.concatenate([epA,eAo]),np.concatenate([epP,ePo]),np.concatenate([epR,eRo])
    Xe=Xs.loc[evm]; XeA,XeP,XeS=mk['A'](Xe,nhs[evm],lhs[evm]),mk['P'](Xe,nhs[evm],lhs[evm]),mk['S'](Xe)
    XeAp,XePp=mk['Aps'](Xe,nhs[evm]),mk['Pps'](Xe,nhs[evm])
    ma=lgbc().fit(XtA,ytA); mp=lgbc().fit(XtP,ytP); mr=lgbc(False).fit(XtS,ytR)
    tA=ManyClassClassifier(estimator=TabPFNClassifier(device=DEV,ignore_pretraining_limits=True),alphabet_size=10,random_state=SEED).fit(XpA,epA)
    tP=TabPFNClassifier(device=DEV,ignore_pretraining_limits=True).fit(XpP,epP); tR=TabPFNClassifier(device=DEV,ignore_pretraining_limits=True).fit(XpS,epR)
    return dict(LA=align(ma.predict_proba(XeA),ma.classes_,ACLS),LP=align(mp.predict_proba(XeP),mp.classes_,PCLS),LR=mr.predict_proba(XeS)[:,1],
                TA=align(tA.predict_proba(XeAp),tA.classes_,ACLS),TP=align(tP.predict_proba(XePp),tP.classes_,PCLS),TR=tR.predict_proba(XeS)[:,1])

CFG=["e","f","g"]; R={c:{k:[] for k in ["LA","LP","LR","TA","TP","TR","GA","GP","GR"]} for c in CFG}
off=np.cumsum([0]+[int((sf==f).sum()) for f in range(5)])
oof_t0=time.time()
for f in range(5):
    fT0=time.time(); sfe=sf==f; Ce,Ne,Le=Call[sfe],Nall[sfe],Lall[sfe]; sl=slice(off[f],off[f+1])
    trn=tr[tr.match.isin([m for m in M if fo[m]!=f])]
    # (e) LGBM/TabPFN identical to v7 OOF -> reuse cache; multi-seed GRUs on unfiltered fold
    for k,src in [("LA","LAc"),("LP","LPc"),("LR","LRc"),("TA","TAc"),("TP","TPc"),("TR","TRc")]: R["e"][k].append(v7[src][sl])
    Cf,Nf,Lf,a_,p_,r_=build_seq(trn,"all",tld)
    for k,v in zip(["GA","GP","GR"],blend_gru(Cf,Nf,Lf,a_,p_,r_,Ce,Ne,Le)): R["e"][k].append(v)
    # (f) noise filter on training folds
    mk=fold_makers(f,keepA)
    for k,v in tab_lgbm_tabpfn(f,mk,keepA,keepS,False).items(): R["f"][k].append(v)
    Cf,Nf,Lf,a_,p_,r_=build_seq(trn,"all",tld,exclude_uids=EXCLUDE_UIDS)
    for k,v in zip(["GA","GP","GR"],blend_gru(Cf,Nf,Lf,a_,p_,r_,Ce,Ne,Le)): R["f"][k].append(v)
    # (g) + old samples for all tasks
    for k,v in tab_lgbm_tabpfn(f,mk,keepA,keepS,True).items(): R["g"][k].append(v)
    g_=blend_gru(np.concatenate([Cf,Ca_o]),np.concatenate([Nf,Na_o]),np.concatenate([Lf,La_o]),np.concatenate([a_,gyA_o]),np.concatenate([p_,gyP_o]),np.concatenate([r_,gyR_o]),Ce,Ne,Le)
    for k,v in zip(["GA","GP","GR"],g_): R["g"][k].append(v)
    out(f"  fold {f} done in {time.time()-fT0:.0f}s (eval n={sfe.sum()})")
for c in CFG:
    for k in R[c]: R[c][k]=np.vstack(R[c][k]) if R[c][k][0].ndim==2 else np.concatenate(R[c][k])
R["h"]={k:(R["g"][k] if k in ("LA","TA","GA") else R["f"][k]) for k in R["g"]}
out(f"OOF ablation complete in {(time.time()-oof_t0)/60:.1f} min")
np.savez('oof_ablation_v10_v14.npz',**{f"{c}_{k}":R[c][k] for c in R for k in R[c]})

# ===== score every configuration with the fixed v9 weights/betas =====
R["d"]=dict(LA=LAc,LP=LPc,LR=LRc,TA=TAc,TP=TPc,TR=TRc,GA=GAc,GP=GPc,GR=GRc)
rows=[]
for c,name in [("d","(d) v9: GRU blend"),("e","(e) + multi-seed"),("f","(e) + noise filter"),("g","(f) + old samples, all tasks [v14]"),("h","(f) + old samples, action only [v14g]")]:
    r=R[c]; BA_=r["LA"]*WA[0]+r["TA"]*WA[1]+r["GA"]*WA[2]; BP_=r["LP"]*WP[0]+r["TP"]*WP[1]+r["GP"]*WP[2]; BR_=r["LR"]*WR[0]+r["TR"]*WR[1]+r["GR"]*WR[2]
    fa=0.94*f1m(BA_,YA,ACLS,prA,BA,SEEN)+0.06*f1m(BA_,YA,ACLS,prA,BA,~SEEN); fp=0.94*f1m(BP_,YP,PCLS,prP,BP,SEEN)+0.06*f1m(BP_,YP,PCLS,prP,BP,~SEEN)
    auc=roc_auc_score(YR,BR_); row=dict(config=name,cvb=0.4*fa+0.4*fp+0.2*auc,action_f1=fa,point_f1=fp,server_auc=auc)
    for sname,msk in [("S",SEEN),("R",RES),("C",COLD)]:
        row[f"act_{sname}"]=f1m(BA_,YA,ACLS,prA,BA,msk); row[f"pt_{sname}"]=f1m(BP_,YP,PCLS,prP,BP,msk)
    rows.append(row)
res=pd.DataFrame(rows); pd.set_option("display.width",250)
out(res.round(4).to_string(index=False))
os.makedirs("../paper_analysis/results",exist_ok=True); res.to_csv("../paper_analysis/results/oof_ablation.csv",index=False)
print("OOFABLATIONDONE")
