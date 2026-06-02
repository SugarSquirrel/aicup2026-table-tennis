"""v9 = v7 features + (0.5 * v7-GRU + 0.5 * v8-GRU) blend at predict time.
Reuses v7 OOF + v8 OOF caches for ensemble weight/beta search.
At test: retrain LGBM + TabPFN + BOTH GRUs (v7+v8) on full train; blend GRU preds.
Expected: action +0.0012, point +0.0011 CV-B over v7 (per OOF blend diagnostic).
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
def build(df,mode,tld,seed=SEED,test_mode=False):
    rng=np.random.default_rng(seed)
    if mode=="sampled": Ls=np.array(sorted(tld)); Ps=np.array([tld[l] for l in Ls],float); Ps/=Ps.sum()
    rows,yA,yP,yR,nh,lh,uid,g=[],[],[],[],[],[],[],[]
    for rid,grp in df.groupby("rally_uid",sort=False):
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
def build_seq(df,mode,tld,seed=SEED,test_mode=False):
    rng=np.random.default_rng(seed)
    if mode=="sampled": Ls=np.array(sorted(tld));Ps=np.array([tld[l] for l in Ls],float);Ps/=Ps.sum()
    C=[];Nu=[];Ln=[];yA=[];yP=[];yR=[]
    for _,grp in df.groupby("rally_uid",sort=False):
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
out(f"v9 (GRU blend α={GRU_BLEND}) CV-B: action F1={fa_b:.4f} w={WA} β={BA:.3f}  | CV-A={fa_a:.4f}")
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
Xa,yA,yP,yR,nha,lha,ga=build(tr,"all",tld); Xs,eA,eP,eR,nhs,lhs,gs=build(tr,"sampled",tld); Xao,yAo,yPo,yRo,nhao,lhao,gao=build(old,"all",tld)
Xte,nht,lht,uids=build(te,"sampled",tld,test_mode=True); t0=time.time()
KEY=['_la','_lp']; BASE=[c for c in Xa.columns if c not in KEY]
# AUG stats (action) + TRAIN-ONLY stats (point + server) — v7 task gating
F_la=np.concatenate([Xa["_la"].to_numpy(),Xao["_la"].to_numpy()]); F_lp=np.concatenate([Xa["_lp"].to_numpy(),Xao["_lp"].to_numpy()])
F_yA=np.concatenate([yA,yAo]); F_yP=np.concatenate([yP,yPo]); F_nh=np.concatenate([nha,nhao]); F_lh=np.concatenate([lha,lhao])
T_AUG=fit_trans({"_la":F_la,"_lp":F_lp},F_yA,F_yP); dMa_A,gA_A,dMp_A,gP_A=player_dists(F_nh,F_yA,F_yP)
clA=fit_clusters(pd.concat([tr,old],ignore_index=True)); cAA,cPA=fit_matchup(F_nh,F_lh,F_yA,F_yP,clA)
T_TO=fit_trans({"_la":Xa["_la"].to_numpy(),"_lp":Xa["_lp"].to_numpy()},yA,yP); dMa_T,gA_T,dMp_T,gP_T=player_dists(nha,yA,yP)
clT=fit_clusters(tr); cAT,cPT=fit_matchup(nha,lha,yA,yP,clT)
def mkA_aug(Xb,nh,lh,idx):
    Ft=apply_trans({k:Xb[k].to_numpy() for k in KEY},T_AUG); Ft.index=idx
    return pd.concat([Xb[BASE],Ft,player_feat(nh,dMa_A,gA_A,dMp_A,gP_A,idx),matchup_feat(nh,lh,clA,cAA,cPA,dMa_A,gA_A,dMp_A,gP_A,idx)],axis=1)
def mkP_to(Xb,nh,lh,idx):
    Ft=apply_trans({k:Xb[k].to_numpy() for k in KEY},T_TO); Ft.index=idx
    return pd.concat([Xb[BASE],Ft,player_feat(nh,dMa_T,gA_T,dMp_T,gP_T,idx),matchup_feat(nh,lh,clT,cAT,cPT,dMa_T,gA_T,dMp_T,gP_T,idx)],axis=1)
def mkAps_aug(Xb,nh,idx):
    Ft=apply_trans({k:Xb[k].to_numpy() for k in KEY},T_AUG); Ft.index=idx
    return pd.concat([Xb[BASE],Ft,player_feat(nh,dMa_A,gA_A,dMp_A,gP_A,idx)],axis=1)
def mkAps_to(Xb,nh,idx):
    Ft=apply_trans({k:Xb[k].to_numpy() for k in KEY},T_TO); Ft.index=idx
    return pd.concat([Xb[BASE],Ft,player_feat(nh,dMa_T,gA_T,dMp_T,gP_T,idx)],axis=1)
def mkS_to(Xb,idx):
    Ft=apply_trans({k:Xb[k].to_numpy() for k in KEY},T_TO); Ft.index=idx
    return pd.concat([Xb[BASE],Ft],axis=1)
XaA=mkA_aug(Xa,nha,lha,Xa.index); XteA=mkA_aug(Xte,nht,lht,Xte.index)
XsAa=mkAps_aug(Xs,nhs,Xs.index); XteAa=mkAps_aug(Xte,nht,Xte.index)
XaP=mkP_to(Xa,nha,lha,Xa.index); XteP=mkP_to(Xte,nht,lht,Xte.index)
XsPt=mkAps_to(Xs,nhs,Xs.index); XtePt=mkAps_to(Xte,nht,Xte.index)
XaS=mkS_to(Xa,Xa.index); XsS=mkS_to(Xs,Xs.index); XteS=mkS_to(Xte,Xte.index)
out("Training LGBM + TabPFN + BOTH GRUs (v7 + v8) on full train...")
mA=lgbc().fit(XaA,yA); mP=lgbc().fit(XaP,yP); mR=lgbc(False).fit(XaS,yR)
tA=ManyClassClassifier(estimator=TabPFNClassifier(device=DEV,ignore_pretraining_limits=True),alphabet_size=10,random_state=SEED).fit(XsAa,eA)
tP=TabPFNClassifier(device=DEV,ignore_pretraining_limits=True).fit(XsPt,eP)
tR=TabPFNClassifier(device=DEV,ignore_pretraining_limits=True).fit(XsS,eR)
Ca,Na,La_,gyA,gyP,gyR=build_seq(tr,"all",tld); Cte,Nte,Lte=build_seq(te,"sampled",tld,test_mode=True)
gm_v7=gru_train(Ca,Na,La_,gyA,gyP,gyR,Cls=GRUNetV7); gm_v8=gru_train(Ca,Na,La_,gyA,gyP,gyR,Cls=GRUNetV8)
gA_v7,gP_v7,gR_v7=gru_pred(gm_v7,Cte,Nte,Lte); gA_v8,gP_v8,gR_v8=gru_pred(gm_v8,Cte,Nte,Lte)
gA_=GRU_BLEND*gA_v7+(1-GRU_BLEND)*gA_v8; gP_=GRU_BLEND*gP_v7+(1-GRU_BLEND)*gP_v8; gR_=GRU_BLEND*gR_v7+(1-GRU_BLEND)*gR_v8
out(f"GRU blend α={GRU_BLEND}: v7 GRU + v8 GRU averaged at test")
PA=WA[0]*align(mA.predict_proba(XteA),mA.classes_,ACLS)+WA[1]*align(tA.predict_proba(XteAa),tA.classes_,ACLS)+WA[2]*gA_
PP=WP[0]*align(mP.predict_proba(XteP),mP.classes_,PCLS)+WP[1]*align(tP.predict_proba(XtePt),tP.classes_,PCLS)+WP[2]*gP_
PR=WR[0]*mR.predict_proba(XteS)[:,1]+WR[1]*tR.predict_proba(XteS)[:,1]+WR[2]*gR_
sgp_true=old.groupby('rally_uid').serverGetPoint.first().to_dict()
PR_ovr=PR.copy(); n_ovr=0
for i,u in enumerate(uids):
    if int(u) in sgp_true: PR_ovr[i]=float(sgp_true[int(u)]); n_ovr+=1
out(f"server override applied to {n_ovr}/{len(uids)} rallies")
def decide(p,cls,pr,b,mask0):
    adj=p/np.clip(pr,1e-9,None)**b
    if mask0: adj=adj.copy(); adj[:,0]=-1e18
    return cls[adj.argmax(1)]
for tag,mask in [("incl0",False),("excl0",True)]:
    aid=decide(PA,ACLS,prA,BA,mask).astype(int); pid=decide(PP,PCLS,prP,BP,mask).astype(int)
    for sfx,pr_ in [("aug",PR),("aug-ovr",PR_ovr)]:
        sub=pd.DataFrame({"rally_uid":uids,"actionId":aid,"pointId":pid,"serverGetPoint":pr_}).sort_values("rally_uid")
        assert len(sub)==len(set(uids)) and not sub.rally_uid.duplicated().any() and sub.serverGetPoint.between(0,1).all()
        sub.to_csv(f"../submission_v9-{sfx}_{tag}.csv",index=False)
    out(f"[{tag}] rows={len(sub)} action={dict(pd.Series(aid).value_counts().head(3))} point={dict(pd.Series(pid).value_counts().head(3))}")
out(f"v9 submissions written {time.time()-t0:.0f}s")
print("V9DONE")
