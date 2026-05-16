"""
SaaS Customer Churn & Retention Analysis — Full Visualization Script
====================================================================
Author  : Data Analytics Team
Dataset : 2,000 synthetic SaaS customers (2023–2024)
Purpose : Reproduce both dashboard figures from the churn analysis report.

Usage
-----
    pip install pandas numpy matplotlib seaborn scipy
    python churn_analysis_visualization.py

Outputs
-------
    churn_dashboard.png  — Main 10-panel KPI dashboard
    churn_deepdive.png   — Deep-dive cohort & revenue dashboard

Dataset columns expected
------------------------
    customer_id, signup_date, plan, region, channel, industry,
    team_size, mrr, feature_usage_score, logins_per_month,
    support_tickets, nps_score, integrations_used,
    churned (0/1), churn_date, churn_reason, tenure_days
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ── Optional: generate synthetic dataset if CSV not found ──────────────────────
import os, random
from datetime import datetime, timedelta

CSV_PATH = "churn_data.csv"

def generate_dataset(n=2000, seed=42):
    """Generate a realistic synthetic SaaS churn dataset."""
    np.random.seed(seed); random.seed(seed)

    plans  = ["Free","Starter","Pro","Enterprise"]
    p_plan = [0.30, 0.35, 0.25, 0.10]
    regions= ["North America","Europe","Asia Pacific","Latin America","Middle East & Africa"]
    p_reg  = [0.38, 0.28, 0.20, 0.09, 0.05]
    channels = ["Organic","Paid Search","Referral","Social","Direct"]
    p_ch   = [0.30, 0.25, 0.20, 0.15, 0.10]
    industries = ["SaaS","E-commerce","Finance","Healthcare","Education","Media","Logistics"]
    churn_base = {"Free":0.65,"Starter":0.42,"Pro":0.22,"Enterprise":0.08}
    mrr_map    = {"Free":0,"Starter":49,"Pro":149,"Enterprise":599}

    start, end = datetime(2023,1,1), datetime(2024,12,31)

    def rdate(s,e):
        return s + timedelta(days=random.randint(0,(e-s).days))

    signups  = [rdate(start, end-timedelta(days=30)) for _ in range(n)]
    plan_arr = np.random.choice(plans, n, p=p_plan)
    reg_arr  = np.random.choice(regions, n, p=p_reg)
    ch_arr   = np.random.choice(channels, n, p=p_ch)
    ind_arr  = np.random.choice(industries, n)
    feat_use = np.random.beta(2,5,n)*100
    logins   = np.random.poisson(8,n)
    tickets  = np.random.poisson(1.5,n)
    nps      = np.random.choice(range(11),n,p=[.04,.03,.04,.05,.06,.08,.10,.14,.18,.16,.12])
    integs   = np.random.poisson(2,n)
    teams    = np.random.choice(range(1,51),n)

    rows=[]
    for i in range(n):
        pl = plan_arr[i]
        pc = churn_base[pl]
        if feat_use[i]>60:   pc*=0.6
        if logins[i]>12:     pc*=0.7
        if nps[i]>=9:        pc*=0.5
        if nps[i]<=3:        pc*=1.5
        if tickets[i]>4:     pc*=1.3
        if integs[i]>=3:     pc*=0.7
        if teams[i]>10:      pc*=0.8
        pc = min(max(pc,0.02),0.95)
        churned = np.random.random()<pc
        sg = signups[i]
        if churned:
            td = int(np.random.exponential(90))
            td = min(td,(end-sg).days)
            cd = sg+timedelta(days=td)
            cr = np.random.choice(
                ["Price too high","Missing features","Switched to competitor",
                 "No longer needed","Poor support","Complexity"],
                p=[0.28,0.22,0.20,0.15,0.10,0.05])
        else:
            td = (end-sg).days; cd=None; cr=None
        mrr = mrr_map[pl]*(1+np.random.normal(0,0.05))
        rows.append(dict(customer_id=f"CUST-{10000+i}",
                         signup_date=sg.strftime("%Y-%m-%d"),
                         plan=pl, region=reg_arr[i], channel=ch_arr[i],
                         industry=ind_arr[i], team_size=int(teams[i]),
                         mrr=round(max(mrr,0),2),
                         feature_usage_score=round(feat_use[i],1),
                         logins_per_month=int(logins[i]),
                         support_tickets=int(tickets[i]),
                         nps_score=int(nps[i]),
                         integrations_used=int(integs[i]),
                         churned=int(churned),
                         churn_date=cd.strftime("%Y-%m-%d") if cd else None,
                         churn_reason=cr, tenure_days=td))
    return pd.DataFrame(rows)

if not os.path.exists(CSV_PATH):
    print(f"'{CSV_PATH}' not found — generating synthetic dataset...")
    df_gen = generate_dataset()
    df_gen.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df_gen):,} rows to {CSV_PATH}")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["signup_date"] = pd.to_datetime(df["signup_date"])
df["churn_date"]  = pd.to_datetime(df["churn_date"])

# ── Palette ────────────────────────────────────────────────────────────────────
BG,PANEL,CARD,BORDER = "#0D1117","#161B22","#21262D","#30363D"
A1,A2,A3,A4,A5       = "#00D4AA","#FF6B6B","#FFA500","#7C6FCD","#3DA9FC"
TEXT,SUB              = "#E6EDF3","#8B949E"
PC = {"Free":"#8B949E","Starter":A5,"Pro":A3,"Enterprise":A4}

BASE_STYLE = dict(
    figure_facecolor=BG, axes_facecolor=PANEL, axes_edgecolor=BORDER,
    axes_labelcolor=TEXT, axes_titlecolor=TEXT,
    xtick_color=SUB, ytick_color=SUB, text_color=TEXT,
    grid_color=CARD,
)
plt.rcParams.update({k.replace("_","."): v for k,v in BASE_STYLE.items()})
plt.rcParams.update({"font.family":"DejaVu Sans",
                      "axes.spines.top":False,"axes.spines.right":False})

# ── Pre-compute metrics ────────────────────────────────────────────────────────
overall_churn    = df["churned"].mean()
avg_tenure       = df[df["churned"]==1]["tenure_days"].mean()
avg_ltv_retained = df[df["churned"]==0]["mrr"].mean()*24
churn_by_plan    = df.groupby("plan")["churned"].mean().reindex(["Free","Starter","Pro","Enterprise"])
churn_by_region  = df.groupby("region")["churned"].mean().sort_values(ascending=False)
churn_by_channel = df.groupby("channel")["churned"].mean().sort_values(ascending=False)
churn_reason_vc  = df[df["churned"]==1]["churn_reason"].value_counts()
churned_df       = df[df["churned"]==1]
retained_df      = df[df["churned"]==0]

nps_bins  = pd.cut(df["nps_score"],[-1,6,8,10],labels=["Detractors\n(0-6)","Passives\n(7-8)","Promoters\n(9-10)"])
nps_churn = df.groupby(nps_bins, observed=True)["churned"].mean()
df["usage_quartile"] = pd.qcut(df["feature_usage_score"],4,labels=["Q1\nLow","Q2","Q3","Q4\nHigh"])
usage_churn = df.groupby("usage_quartile", observed=True)["churned"].mean()

mrr_lost_plan    = df[df["churned"]==1].groupby("plan")["mrr"].sum()
mrr_retained_plan= df[df["churned"]==0].groupby("plan")["mrr"].sum()
monthly_churns   = df[df["churned"]==1].groupby(df["churn_date"].dt.to_period("M")).size()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main KPI Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(26,22), facecolor=BG)
gs_outer = gridspec.GridSpec(4,1,figure=fig1,hspace=0.45,
                              top=0.93,bottom=0.04,left=0.04,right=0.97)

# ── Title + KPI cards ─────────────────────────────────────────────────────────
ax_t = fig1.add_subplot(gs_outer[0])
ax_t.set_facecolor(BG); ax_t.axis("off")
ax_t.text(0,0.75,"SAAS CUSTOMER RETENTION & CHURN ANALYSIS",
          fontsize=28,fontweight="bold",color=TEXT,va="top")
ax_t.text(0,0.28,"Executive Dashboard  •  2023–2024  •  2,000 Customers  •  Multi-Plan, Multi-Region",
          fontsize=13,color=SUB,va="top")

kpis = [("OVERALL CHURN",f"{overall_churn:.1%}",A2),
        ("AVG CHURN TENURE",f"{avg_tenure:.0f} days",A3),
        ("RETENTION RATE",f"{1-overall_churn:.1%}",A1),
        ("AVG RETAINED LTV",f"${avg_ltv_retained:,.0f}",A4),
        ("CUSTOMERS CHURNED",f"{df['churned'].sum():,}",A2),
        ("ACTIVE CUSTOMERS",f"{(df['churned']==0).sum():,}",A1)]
for xi,(label,val,color) in zip(np.linspace(0,1,7)[:-1],kpis):
    rect=FancyBboxPatch((xi,-0.05),0.155,0.30,boxstyle="round,pad=0.01",
                         linewidth=1.5,edgecolor=color,facecolor=CARD,
                         transform=ax_t.transAxes,clip_on=False)
    ax_t.add_patch(rect)
    ax_t.text(xi+0.078,0.20,val,fontsize=18,fontweight="bold",color=color,
              ha="center",va="bottom",transform=ax_t.transAxes)
    ax_t.text(xi+0.078,-0.02,label,fontsize=8,color=SUB,
              ha="center",va="top",transform=ax_t.transAxes)
ax_t.set_xlim(0,1); ax_t.set_ylim(-0.3,1)

# ── Row 1 ─────────────────────────────────────────────────────────────────────
gs1 = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs_outer[1],wspace=0.35)

# Churn by plan
ax1 = fig1.add_subplot(gs1[0])
cb = [PC[p] for p in churn_by_plan.index]
bars=ax1.bar(churn_by_plan.index,churn_by_plan.values*100,color=cb,width=0.6,zorder=3,edgecolor=BG)
for bar,v in zip(bars,churn_by_plan.values):
    ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f"{v:.1%}",
             ha="center",va="bottom",fontsize=11,fontweight="bold",color=TEXT)
ax1.set_title("Churn Rate by Plan",fontsize=14,fontweight="bold",pad=12)
ax1.set_ylabel("Churn Rate (%)",color=SUB,fontsize=10)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.grid(axis="y",alpha=0.3); ax1.set_ylim(0,80)
ax1.tick_params(axis="x",colors=TEXT,labelsize=10)

# Churn reasons
ax2 = fig1.add_subplot(gs1[1])
rp = (churn_reason_vc/churn_reason_vc.sum()*100).sort_values()
rc=[A2 if v==rp.max() else A3 if v>rp.median() else SUB for v in rp.values]
hb=ax2.barh(rp.index,rp.values,color=rc,height=0.6,zorder=3,edgecolor=BG)
for bar,v in zip(hb,rp.values):
    ax2.text(v+0.3,bar.get_y()+bar.get_height()/2,f"{v:.1f}%",
             va="center",fontsize=10,color=TEXT,fontweight="bold")
ax2.set_title("Top Churn Reasons",fontsize=14,fontweight="bold",pad=12)
ax2.set_xlabel("% of Churned Customers",color=SUB,fontsize=10)
ax2.grid(axis="x",alpha=0.3); ax2.set_xlim(0,rp.max()*1.25)

# Tenure distribution
ax3 = fig1.add_subplot(gs1[2])
ax3.hist(churned_df["tenure_days"].clip(0,365),bins=30,alpha=0.75,
         color=A2,label="Churned",density=True,zorder=3)
ax3.hist(retained_df["tenure_days"].clip(0,730),bins=30,alpha=0.6,
         color=A1,label="Retained",density=True,zorder=2)
ax3.axvline(churned_df["tenure_days"].median(),color=A2,ls="--",lw=2,
            label=f"Median Churn: {churned_df['tenure_days'].median():.0f}d")
ax3.set_title("Tenure Distribution",fontsize=14,fontweight="bold",pad=12)
ax3.set_xlabel("Tenure (Days)",color=SUB,fontsize=10)
ax3.set_ylabel("Density",color=SUB,fontsize=10)
ax3.legend(fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
ax3.grid(axis="y",alpha=0.3)

# ── Row 2 ─────────────────────────────────────────────────────────────────────
gs2 = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs_outer[2],wspace=0.35)

# Region
ax4 = fig1.add_subplot(gs2[0])
rc2=[A2 if v==churn_by_region.max() else A3 if v>churn_by_region.mean() else A1 for v in churn_by_region.values]
hb2=ax4.barh(churn_by_region.index,churn_by_region.values*100,color=rc2,height=0.55,zorder=3,edgecolor=BG)
for bar,v in zip(hb2,churn_by_region.values):
    ax4.text(v*100+0.2,bar.get_y()+bar.get_height()/2,f"{v:.1%}",
             va="center",fontsize=10,color=TEXT,fontweight="bold")
ax4.set_title("Churn Rate by Region",fontsize=14,fontweight="bold",pad=12)
ax4.set_xlabel("Churn Rate (%)",color=SUB,fontsize=10)
ax4.xaxis.set_major_formatter(mtick.PercentFormatter())
ax4.grid(axis="x",alpha=0.3); ax4.set_xlim(0,churn_by_region.max()*130)

# NPS vs churn
ax5 = fig1.add_subplot(gs2[1])
nb=ax5.bar(nps_churn.index,nps_churn.values*100,color=[A2,A3,A1],width=0.55,zorder=3,edgecolor=BG)
for bar,v in zip(nb,nps_churn.values):
    ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f"{v:.1%}",
             ha="center",va="bottom",fontsize=12,fontweight="bold",color=TEXT)
ax5.set_title("Churn Rate by NPS Group",fontsize=14,fontweight="bold",pad=12)
ax5.set_ylabel("Churn Rate (%)",color=SUB,fontsize=10)
ax5.yaxis.set_major_formatter(mtick.PercentFormatter())
ax5.grid(axis="y",alpha=0.3); ax5.set_ylim(0,65)
ax5.tick_params(axis="x",colors=TEXT,labelsize=10)

# Feature usage vs churn
ax6 = fig1.add_subplot(gs2[2])
ub=ax6.bar(usage_churn.index,usage_churn.values*100,color=[A2,A3,A3,A1],width=0.55,zorder=3,edgecolor=BG)
for bar,v in zip(ub,usage_churn.values):
    ax6.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f"{v:.1%}",
             ha="center",va="bottom",fontsize=12,fontweight="bold",color=TEXT)
ax6.set_title("Churn Rate by Feature Usage",fontsize=14,fontweight="bold",pad=12)
ax6.set_ylabel("Churn Rate (%)",color=SUB,fontsize=10)
ax6.yaxis.set_major_formatter(mtick.PercentFormatter())
ax6.grid(axis="y",alpha=0.3); ax6.set_ylim(0,60)
ax6.tick_params(axis="x",colors=TEXT,labelsize=10)

# ── Row 3 ─────────────────────────────────────────────────────────────────────
gs3 = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs_outer[3],wspace=0.35)

# MRR retained vs lost
ax7 = fig1.add_subplot(gs3[0])
po=["Free","Starter","Pro","Enterprise"]
ml=mrr_lost_plan.reindex(po).fillna(0); mr=mrr_retained_plan.reindex(po).fillna(0)
xp=np.arange(4); w=0.38
ax7.bar(xp-w/2,mr/1000,width=w,color=A1,label="Retained MRR",zorder=3,edgecolor=BG)
ax7.bar(xp+w/2,ml/1000,width=w,color=A2,label="Lost MRR",    zorder=3,edgecolor=BG)
ax7.set_xticks(xp); ax7.set_xticklabels(po,color=TEXT,fontsize=10)
ax7.set_title("MRR: Retained vs Lost",fontsize=14,fontweight="bold",pad=12)
ax7.set_ylabel("MRR ($K)",color=SUB,fontsize=10)
ax7.legend(fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
ax7.grid(axis="y",alpha=0.3)

# Monthly churn trend
ax8 = fig1.add_subplot(gs3[1])
mc=monthly_churns.reset_index()
mc.columns=["month","churns"]; mc["m"]=mc["month"].astype(str)
mc=mc.sort_values("month")
xr=range(len(mc))
ax8.fill_between(xr,mc["churns"].values,alpha=0.3,color=A2)
ax8.plot(xr,mc["churns"].values,color=A2,lw=2.5,marker="o",ms=5,zorder=3)
roll=mc["churns"].rolling(3,center=True).mean()
ax8.plot(xr,roll.values,color=A3,lw=2,ls="--",label="3M Avg",zorder=4)
step=max(1,len(mc)//8)
ax8.set_xticks(list(xr)[::step])
ax8.set_xticklabels(mc["m"].values[::step],rotation=35,ha="right",fontsize=8,color=SUB)
ax8.set_title("Monthly Churn Volume",fontsize=14,fontweight="bold",pad=12)
ax8.set_ylabel("Customers Churned",color=SUB,fontsize=10)
ax8.legend(fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
ax8.grid(axis="y",alpha=0.3)

# Channel churn
ax9 = fig1.add_subplot(gs3[2])
cc=churn_by_channel.sort_values(ascending=True)
ccs=[A2 if v==cc.max() else A3 if v>cc.mean() else A1 for v in cc.values]
hb3=ax9.barh(cc.index,cc.values*100,color=ccs,height=0.55,zorder=3,edgecolor=BG)
for bar,v in zip(hb3,cc.values):
    ax9.text(v*100+0.2,bar.get_y()+bar.get_height()/2,f"{v:.1%}",
             va="center",fontsize=10,color=TEXT,fontweight="bold")
ax9.set_title("Churn by Acquisition Channel",fontsize=14,fontweight="bold",pad=12)
ax9.set_xlabel("Churn Rate (%)",color=SUB,fontsize=10)
ax9.xaxis.set_major_formatter(mtick.PercentFormatter())
ax9.grid(axis="x",alpha=0.3); ax9.set_xlim(0,cc.max()*130)

fig1.savefig("churn_dashboard.png",dpi=160,bbox_inches="tight",facecolor=BG)
plt.close(fig1)
print("✓ churn_dashboard.png saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Deep-Dive: Cohort, Engagement & Revenue
# ═══════════════════════════════════════════════════════════════════════════════
df["cohort"]     = df["signup_date"].dt.to_period("M")
df["int_group"]  = df["integrations_used"].apply(lambda x:"0" if x==0 else "1-2" if x<=2 else "3+")
df["login_group"]= df["logins_per_month"].apply(
    lambda x:"Low(<5)" if x<5 else "Mid(5-12)" if x<=12 else "High(12+)")

cohort_pivot = df.groupby("cohort")["churned"].mean().reset_index()
cohort_pivot["cs"] = cohort_pivot["cohort"].astype(str)
cohort_pivot = cohort_pivot.sort_values("cohort")

heat_data = df.groupby(["int_group","login_group"])["churned"].mean().unstack()
heat_data = heat_data.reindex(["0","1-2","3+"]).reindex(
    columns=["Low(<5)","Mid(5-12)","High(12+)"])

bl=[-1,30,60,90,180,365,9999]
bla=["0-30d","31-60d","61-90d","91-180d","181-365d","365d+"]
churned_df2=df[df["churned"]==1].copy()
churned_df2["bucket"]=pd.cut(churned_df2["tenure_days"],bins=bl,labels=bla,right=True)
surv=churned_df2.groupby(["bucket","plan"],observed=True).size().unstack(fill_value=0)
surv_pct=surv.div(surv.sum(axis=0),axis=1)*100

sample=df.sample(400,random_state=42)
plan_mrr=df[df["churned"]==0].groupby("plan")["mrr"].sum().reindex(["Free","Starter","Pro","Enterprise"])
lost_mrr=df[df["churned"]==1]["mrr"].sum()
by_plan=df.groupby("plan")["churned"].agg(["count","sum"]).reindex(["Free","Starter","Pro","Enterprise"])

fig2=plt.figure(figsize=(26,20),facecolor=BG)

ax_t2=plt.axes([0.0,0.92,1.0,0.08]); ax_t2.set_facecolor(BG); ax_t2.axis("off")
ax_t2.text(0.03,0.75,"DEEP-DIVE: COHORT, ENGAGEMENT & REVENUE ANALYSIS",
           fontsize=24,fontweight="bold",color=TEXT,va="top",transform=ax_t2.transAxes)
ax_t2.text(0.03,0.20,"Retention Drivers  •  Engagement Heatmap  •  Plan Economics  •  MRR Recovery",
           fontsize=12,color=SUB,va="top",transform=ax_t2.transAxes)

# Monthly cohort churn
ax_c=plt.axes([0.05,0.65,0.27,0.24]); ax_c.set_facecolor(PANEL)
months=cohort_pivot["cs"].values; rates=cohort_pivot["churned"].values*100
ax_c.fill_between(range(len(months)),rates,alpha=0.18,color=A2)
ax_c.plot(range(len(months)),rates,color=A2,lw=2.5,marker="o",ms=4,zorder=4)
roll3=pd.Series(rates).rolling(3,center=True,min_periods=1).mean().values
ax_c.plot(range(len(months)),roll3,color=A3,lw=2,ls="--",label="3M Avg")
step=max(1,len(months)//7)
ax_c.set_xticks(range(0,len(months),step))
ax_c.set_xticklabels(months[::step],rotation=35,ha="right",fontsize=8)
ax_c.set_title("Monthly Cohort Churn Rate",fontsize=13,fontweight="bold",pad=10)
ax_c.set_ylabel("Churn Rate (%)",fontsize=9,color=SUB)
ax_c.yaxis.set_major_formatter(mtick.PercentFormatter())
ax_c.legend(fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
ax_c.grid(axis="y",alpha=0.3)
for sp in ["top","right"]: ax_c.spines[sp].set_visible(False)

# Engagement heatmap
ax_h=plt.axes([0.38,0.65,0.25,0.24]); ax_h.set_facecolor(PANEL)
cmap_rg=mcolors.LinearSegmentedColormap.from_list("rg",[A1,A3,A2])
im=ax_h.imshow(heat_data.values,cmap=cmap_rg,aspect="auto",vmin=0,vmax=0.6)
ax_h.set_xticks(range(len(heat_data.columns))); ax_h.set_xticklabels(heat_data.columns,fontsize=9,color=TEXT)
ax_h.set_yticks(range(len(heat_data.index)));   ax_h.set_yticklabels(heat_data.index,fontsize=9,color=TEXT)
ax_h.set_xlabel("Login Frequency",fontsize=9,color=SUB)
ax_h.set_ylabel("Integrations Used",fontsize=9,color=SUB)
ax_h.set_title("Churn Heatmap: Integrations × Logins",fontsize=13,fontweight="bold",pad=10)
for i in range(len(heat_data.index)):
    for j in range(len(heat_data.columns)):
        v=heat_data.values[i,j]
        ax_h.text(j,i,f"{v:.1%}",ha="center",va="center",
                  fontsize=12,fontweight="bold",color="white" if v>0.3 else TEXT)
plt.colorbar(im,ax=ax_h,fraction=0.04,pad=0.04).ax.tick_params(colors=SUB,labelsize=8)

# Time-to-churn stacked bar
ax_s=plt.axes([0.70,0.65,0.27,0.24]); ax_s.set_facecolor(PANEL)
bcols=[A2,"#FF8E53","#FFB347",A3,"#8BC34A",A1]
po=["Free","Starter","Pro","Enterprise"]; bot_s=np.zeros(4)
for bkt,color in zip(surv_pct.index,bcols):
    vals=surv_pct.loc[bkt].reindex(po).fillna(0).values
    ax_s.bar(po,vals,bottom=bot_s,color=color,label=bkt,edgecolor=BG,lw=0.5)
    bot_s+=vals
ax_s.set_title("When Do Customers Churn?",fontsize=13,fontweight="bold",pad=10)
ax_s.set_ylabel("% of Churned",fontsize=9,color=SUB)
ax_s.yaxis.set_major_formatter(mtick.PercentFormatter())
ax_s.legend(fontsize=8,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT,loc="upper right",ncol=2)
ax_s.grid(axis="y",alpha=0.3); ax_s.set_ylim(0,115); ax_s.tick_params(axis="x",colors=TEXT)
for sp in ["top","right"]: ax_s.spines[sp].set_visible(False)

# Scatter logins vs tenure
ax_sc=plt.axes([0.05,0.35,0.27,0.24]); ax_sc.set_facecolor(PANEL)
cs2=sample[sample["churned"]==1]; rs2=sample[sample["churned"]==0]
ax_sc.scatter(rs2["logins_per_month"].clip(0,25),rs2["tenure_days"].clip(0,600),
              alpha=0.45,s=22,color=A1,label="Retained",zorder=3)
ax_sc.scatter(cs2["logins_per_month"].clip(0,25),cs2["tenure_days"].clip(0,600),
              alpha=0.45,s=22,color=A2,label="Churned",zorder=4)
ax_sc.set_title("Logins / Month vs Tenure",fontsize=13,fontweight="bold",pad=10)
ax_sc.set_xlabel("Logins per Month",fontsize=9,color=SUB)
ax_sc.set_ylabel("Tenure (Days)",fontsize=9,color=SUB)
ax_sc.legend(fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
ax_sc.grid(alpha=0.2)
for sp in ["top","right"]: ax_sc.spines[sp].set_visible(False)

# Active MRR donut
ax_d=plt.axes([0.38,0.35,0.22,0.24]); ax_d.set_facecolor(BG); ax_d.axis("off")
wedges,_,ats=ax_d.pie(plan_mrr.fillna(0).values,labels=plan_mrr.index,
    colors=[PC[p] for p in plan_mrr.index],autopct="%1.1f%%",startangle=90,
    wedgeprops={"edgecolor":BG,"linewidth":2},
    textprops={"color":TEXT,"fontsize":10},pctdistance=0.78)
for at in ats: at.set_fontweight("bold"); at.set_fontsize(9)
ax_d.add_patch(plt.Circle((0,0),0.55,fc=BG))
ax_d.text(0,0.08,"ACTIVE",ha="center",fontsize=10,color=SUB,fontweight="bold")
ax_d.text(0,-0.12,"MRR",ha="center",fontsize=10,color=SUB,fontweight="bold")
ax_d.set_title("Active MRR by Plan",fontsize=13,fontweight="bold",color=TEXT,pad=20)

# Feature usage boxplot
ax_f=plt.axes([0.66,0.35,0.30,0.24]); ax_f.set_facecolor(PANEL)
for pos,plan in zip([1,2,3,4],["Free","Starter","Pro","Enterprise"]):
    ax_f.boxplot(df[df["plan"]==plan]["feature_usage_score"].values,
                 positions=[pos],widths=0.5,patch_artist=True,
                 medianprops={"color":BG,"lw":2.5},
                 boxprops={"facecolor":PC[plan],"edgecolor":BORDER,"alpha":0.85},
                 whiskerprops={"color":SUB,"lw":1.5},capprops={"color":SUB,"lw":2},
                 flierprops={"marker":"o","markerfacecolor":PC[plan],"markersize":3,
                             "alpha":0.4,"markeredgewidth":0})
ax_f.set_xticks([1,2,3,4]); ax_f.set_xticklabels(["Free","Starter","Pro","Enterprise"],
                                                    color=TEXT,fontsize=10)
ax_f.set_title("Feature Usage Score by Plan",fontsize=13,fontweight="bold",pad=10)
ax_f.set_ylabel("Usage Score (0-100)",fontsize=9,color=SUB)
ax_f.grid(axis="y",alpha=0.3)
for sp in ["top","right"]: ax_f.spines[sp].set_visible(False)

# Retained vs churned by plan
ax_w=plt.axes([0.05,0.05,0.40,0.24]); ax_w.set_facecolor(PANEL)
xp=np.arange(4); w=0.38
ax_w.bar(xp-w/2,by_plan["count"]-by_plan["sum"],width=w,color=A1,label="Retained",edgecolor=BG,zorder=3)
ax_w.bar(xp+w/2,by_plan["sum"],                 width=w,color=A2,label="Churned", edgecolor=BG,zorder=3)
for i,plan in enumerate(["Free","Starter","Pro","Enterprise"]):
    rate=by_plan.loc[plan,"sum"]/by_plan.loc[plan,"count"]
    ax_w.text(i,by_plan.loc[plan,"count"]+8,f"{rate:.0%} churned",
              ha="center",fontsize=9,color=TEXT,fontweight="bold")
ax_w.set_xticks(xp); ax_w.set_xticklabels(["Free","Starter","Pro","Enterprise"],
                                             color=TEXT,fontsize=11)
ax_w.set_title("Retained vs Churned by Plan",fontsize=13,fontweight="bold",pad=10)
ax_w.set_ylabel("Customer Count",fontsize=9,color=SUB)
ax_w.legend(fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
ax_w.grid(axis="y",alpha=0.3)
for sp in ["top","right"]: ax_w.spines[sp].set_visible(False)

# MRR recovery potential
ax_r=plt.axes([0.54,0.05,0.42,0.24]); ax_r.set_facecolor(PANEL)
scens=["Current\nLost MRR","Recover 25%","Recover 50%","Recover 75%"]
rvals=[lost_mrr,lost_mrr*0.75,lost_mrr*0.50,lost_mrr*0.25]
rb=ax_r.bar(range(4),[v/1000 for v in rvals],color=[A2,A3,A3,A1],width=0.55,zorder=3,edgecolor=BG)
for bar,v in zip(rb,rvals):
    ax_r.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
              f"${v/1000:.1f}K",ha="center",va="bottom",fontsize=11,color=TEXT,fontweight="bold")
ax_r.set_xticks(range(4)); ax_r.set_xticklabels(scens,color=TEXT,fontsize=9)
ax_r.set_title("MRR at Risk → Recovery Potential",fontsize=13,fontweight="bold",pad=10)
ax_r.set_ylabel("MRR ($K)",fontsize=9,color=SUB)
ax_r.grid(axis="y",alpha=0.3)
for sp in ["top","right"]: ax_r.spines[sp].set_visible(False)

fig2.savefig("churn_deepdive.png",dpi=160,bbox_inches="tight",facecolor=BG)
plt.close(fig2)
print("✓ churn_deepdive.png saved")
print("\nAll done! Open churn_dashboard.png and churn_deepdive.png to view results.")
