import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['mathtext.default'] = 'regular'   # upright math, like the surrounding text
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
HERE = os.path.dirname(os.path.abspath(__file__))
BL,OR,GR,GY,RD='#2a5a9a','#b5651d','#1a7f37','#8a8f98','#b03030'

# ---------------- fig 1: the 1-D picture of route A (conditional quantile map)
D={'MC, context c (L1 hit)':(-2.62,0.10,BL),
   "MC, context c′ (no L1)":(-2.38,0.12,OR),
   "data, context c′ (no L1)":(-2.31,0.14,GR)}
q=0.30; x=np.linspace(-3.0,-1.85,600)
fig,ax=plt.subplots(3,1,figsize=(9,8.6),gridspec_kw=dict(height_ratios=[1,1.25,1.25],hspace=0.42))
z=np.linspace(-3.5,3.5,400); u=norm.ppf(q)
ax[0].plot(z,norm.pdf(z),color='k'); ax[0].fill_between(z[z<=u],norm.pdf(z[z<=u]),color='#ddd')
ax[0].axvline(u,color=RD,ls='--'); ax[0].plot(u,norm.pdf(u),'o',color=RD,ms=8)
ax[0].set_title('latent space: u ~ N(0, 1), the same for every flow and every context',fontsize=11)
ax[0].text(-3.45,0.27,'u = %.2f\n(30%% of the area to the left)'%u,color=RD,fontsize=10)
ax[0].set_xlabel('u'); ax[0].set_yticks([])
pts={}
for lab,(m,s,c) in D.items():
    ax[1].plot(x,norm.pdf(x,m,s),color=c,lw=2,label=lab)
    ax[2].plot(x,norm.cdf(x,m,s),color=c,lw=2)
    pts[lab]=norm.ppf(q,m,s)
ax[2].axhline(q,color=RD,ls='--',lw=1)
for lab,(m,s,c) in D.items():
    xv=pts[lab]; ax[2].plot(xv,q,'o',color=c,ms=9,zorder=5); ax[1].plot(xv,norm.pdf(xv,m,s),'o',color=c,ms=9,zorder=5)
    ax[2].plot([xv,xv],[0,q],color=c,lw=1,ls=':')
k=list(pts)
for a_,b_,col in ((k[0],k[1],OR),(k[1],k[2],GR)):
    ax[2].add_patch(FancyArrowPatch((pts[a_],q),(pts[b_],q),connectionstyle='arc3,rad=-0.35',arrowstyle='-|>',mutation_scale=14,color=col,lw=1.6))
ax[2].text(-2.18,0.42,'orange arrow: the MC flow, c → c′\n    (route A, MC part: C → $C′_{\\mathrm{MC}}$)',color=OR,fontsize=9.5)
ax[2].text(-2.18,0.05,'green arrow: covflow at c′\n    ($C′_{\\mathrm{MC}}$ → C′)',color=GR,fontsize=9.5)
ax[2].text(-2.99,q+0.03,'same quantile: 0.30',color=RD,fontsize=9.5)
ax[1].legend(fontsize=9.5,frameon=False,loc='upper right'); ax[1].set_yticks([])
ax[1].set_title('three conditional distributions of $\\log_{10}$ σ($d_{xy}$) (toy numbers)',fontsize=11)
ax[2].set_title('their cumulative distributions: route A keeps the track at the same height',fontsize=11)
ax[2].set_xlabel('$\\log_{10}$ σ($d_{xy}$) [cm]'); ax[2].set_ylabel('cumulative')
for a in ax[1:]: a.set_xlim(-3.0,-1.85)
fig.savefig(os.path.join(HERE, 'f18_routeA_quantile_1d.png'),dpi=150,bbox_inches='tight'); plt.close(fig)

# ---------------- fig 2: the algebra of route A as a chain
fig,ax=plt.subplots(figsize=(10,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,4.2); ax.axis('off')
def box(x,y,w,h,t,fc='white',ec=BL,tc='k',fs=10.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',fc=fc,ec=ec,lw=1.5))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=tc)
def arr(x0,y0,x1,y1,t,col,dy=0.12,ha='center'):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=13,color=col,lw=1.6))
    ax.text((x0+x1)/2,(y0+y1)/2+dy,t,ha=ha,va='bottom',fontsize=9.5,color=col)
# top row: MC tracks
box(0.1,2.7,1.7,0.9,'C  (MC, with hit)\ncontext c')
box(4.15,2.7,1.7,0.9,'u  (latent)',fc='#f1f1f1',ec='k')
box(8.2,2.7,1.7,0.9,"$C′_{\\mathrm{MC}}$  (MC-like,\nhit lost), c′",ec=OR)
arr(1.85,3.15,4.1,3.15,'$f_{\\mathrm{MC}}$( · ; c)   forward',BL)
arr(5.9,3.15,8.15,3.15,"$f_{\\mathrm{MC}}$⁻¹( · ; c′)   inverse",OR)
box(8.2,0.35,1.7,0.9,"C′  (data-like,\nhit lost), c′",fc=GR,ec=GR,tc='white')
arr(9.05,2.65,9.05,1.3,"covflow at c′:\n$f_{\\mathrm{data}}$⁻¹( $f_{\\mathrm{MC}}$( · ; c′); c′)",GR,dy=0,ha='left')
ax.text(9.15,1.55,'',fontsize=1)
arr(5.0,2.65,8.15,0.85,"$f_{\\mathrm{data}}$⁻¹( · ; c′)   inverse",GR,dy=-0.55)
ax.text(0.1,1.55,"Route A uses ONE latent point u per track.\n\n"
        "MC part (written to the tree): only the MC flow,\nforward at c, inverse at c′.\n\n"
        "Data part: the usual covflow at c′. Since\n$f_{\\mathrm{MC}}$( $f_{\\mathrm{MC}}$⁻¹(u; c′); c′) = u, the two steps\ngive exactly $f_{\\mathrm{data}}$⁻¹( $f_{\\mathrm{MC}}$(C; c); c′ ).",
        fontsize=9.5,va='center',color='#333')
fig.savefig(os.path.join(HERE, 'f19_routeA_chain.png'),dpi=150,bbox_inches='tight'); plt.close(fig)

# ---------------- fig 3: nested fits, toy straight line through 4 BPix layers
rng=np.random.default_rng(1)
r=np.array([2.9,6.8,10.9,16.0]); s=10e-4   # cm, 10 um per hit
X=np.c_[np.ones(4),r]
N=200000
y=rng.normal(0,s,(N,4))                     # true track: a=0, b=0
def fit(Xm,ym):
    Cinv=Xm.T@Xm/s**2; C=np.linalg.inv(Cinv); th=(ym@Xm/s**2)@C; return th,C
thF,CF=fit(X,y); thR,CR=fit(X[1:],y[:,1:])
d=thR-thF
sF,sR=np.sqrt(CF[0,0]),np.sqrt(CR[0,0])
Dm=CR-CF; lam,Q=np.linalg.eigh(Dm)
delta=rng.multivariate_normal([0,0],Dm,N,method='eigh')
fig,ax=plt.subplots(2,2,figsize=(10,8.4),gridspec_kw=dict(hspace=0.42,wspace=0.3))
a=ax[0,0]; sub=slice(0,4000)
a.scatter(thF[sub,0]*1e4,d[sub,0]*1e4,s=3,alpha=0.4,color=BL)
a.set_xlabel('a, full fit (4 hits) [μm]'); a.set_ylabel('a(3 hits) − a(4 hits) [μm]')
a.set_title('(a) the change is independent of the full fit\ncorrelation = %.3f'%np.corrcoef(thF[:,0],d[:,0])[0,1],fontsize=10.5)
a=ax[0,1]; b=np.linspace(-60,60,121)
a.hist(thR[:,0]*1e4,b,histtype='stepfilled',color='#aecde5',label='refit without L1 (truth)')
a.hist((thF[:,0]+delta[:,0])*1e4,b,histtype='step',color=GR,lw=2,ls='--',label='full fit + δ,  δ ~ N(0, $C_{R}$ − C)')
a.hist(thF[:,0]*1e4,b,histtype='step',color=GY,lw=1.5,label='full fit, not smeared')
a.set_xlabel('a = impact parameter [μm]'); a.legend(fontsize=8.5,frameon=False,loc='upper left'); a.set_ylim(0,a.get_ylim()[1]*1.3)
a.set_title('(b) smearing by the difference reproduces the refit\nσ(4 hits) = %.1f μm, σ(3 hits) = %.1f μm'%(sF*1e4,sR*1e4),fontsize=10.5)
a=ax[1,0]; b=np.linspace(-5,5,101)
a.hist(thR[:,0]/sR,b,histtype='stepfilled',color='#aecde5',label='refit / $\\sigma_{R}$: width %.2f'%np.std(thR[:,0]/sR))
a.hist((thF[:,0]+delta[:,0])/sR,b,histtype='step',color=GR,lw=2,ls='--',label='(full + δ) / $\\sigma_{R}$: width %.2f'%np.std((thF[:,0]+delta[:,0])/sR))
a.hist(thF[:,0]/sR,b,histtype='step',color=RD,lw=1.5,label='full / $\\sigma_{R}$, no smearing: width %.2f'%np.std(thF[:,0]/sR))
a.set_xlabel('pull = a / $\\sigma_{R}$'); a.legend(fontsize=8.5,frameon=False,loc='upper left'); a.set_ylim(0,a.get_ylim()[1]*1.45)
a.set_title('(c) enlarging C alone gives pulls that are too narrow',fontsize=10.5)
a=ax[1,1]
def ell(C,col,lab,ls='-'):
    w,v=np.linalg.eigh(C); ang=np.degrees(np.arctan2(v[1,1],v[0,1]))
    a.add_patch(Ellipse((0,0),2*np.sqrt(w[1])*1e4,2*np.sqrt(w[0])*1e4*1e2,angle=0,fill=False))
# draw in (a [um], b [um/cm]) coordinates via sampling a parametric ellipse
def ell2(C,col,lab,ls='-'):
    t=np.linspace(0,2*np.pi,400); L=np.linalg.cholesky(C); p=(L@np.vstack([np.cos(t),np.sin(t)]))
    a.plot(p[0]*1e4,p[1]*1e4,color=col,ls=ls,lw=2,label=lab)
ell2(CF,BL,'C: 4 hits'); ell2(CR,OR,'$C_{R}$: L1 removed')
v=Q[:,1]*np.sqrt(lam[1]); a.annotate('',xy=(v[0]*1e4,v[1]*1e4),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=GR,lw=2))
a.text(-18.5,-0.75,'green arrow: the only\ndirection δ moves in',color=GR,fontsize=9)
a.set_xlabel('a [μm]'); a.set_ylabel('b [μm / cm]'); a.legend(fontsize=8.5,frameon=False,loc='lower right')
a.set_title('(d) one hit = one measurement: $C_{R}$ − C has rank 1,\nits other eigenvalue is exactly 0',fontsize=10.5)
fig.savefig(os.path.join(HERE, 'f20_smearing_nested_fit.png'),dpi=150,bbox_inches='tight'); plt.close(fig)
print('sF %.2f um sR %.2f um ratio %.2f $\\log_{10}$ %.3f; corr %.4f; eig %s'%(sF*1e4,sR*1e4,sR/sF,np.log10(sR/sF),np.corrcoef(thF[:,0],d[:,0])[0,1],lam*1e8))
print('Cov(d) vs CR-CF', np.cov(d.T)*1e8, Dm*1e8)
