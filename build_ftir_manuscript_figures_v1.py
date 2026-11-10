# -*- coding: utf-8 -*-
"""Publication figure pack from existing audited results. No fitting or training."""
import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
BD=ROOT/'results/bidirectional_c4_c8_final'
EP=ROOT/'results/protonet_bidirectional_final'
PC=ROOT/'results/pca128_shrinkage_bidirectional_v1'
CLASSES=['PET','HDPE','PVC','LDPE','PP','PS']
CASES=list('ABCDEF')
LABEL={'raw_1nn':'Spectrum 1-NN','raw_svm':'Spectrum SVM','PCA64_shrinkage':'PCA64 + shrinkage',
       'PCA128_shrinkage':'PCA128 + shrinkage','CE_shrinkage':'CE-M1 + shrinkage','EP_shrinkage':'EP-M1 + shrinkage',
       'EP_euclidean':'EP-M1 + Euclidean','EP_full':'EP-M1 + full Mahalanobis'}
COLOR={'raw_1nn':'#777777','raw_svm':'#141414','PCA64_shrinkage':'#56B4E9','PCA128_shrinkage':'#009E73',
       'CE_shrinkage':'#0072B2','EP_shrinkage':'#D55E00','EP_euclidean':'#CC79A7','EP_full':'#8064A2'}
TASKLABEL={'holdout_hdpe_ldpe':'HDPE / LDPE held out','holdout_pp_ps':'PP / PS held out','holdout_pet_pvc':'PET / PVC held out'}
METRICS=['all_accuracy','all_macro_f1','base_accuracy','seen_novel_accuracy','current_novel_accuracy',
         'previous_novel_accuracy','base_to_novel_error_rate','previous_novel_change_pp']


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


class Pack:
    def __init__(self,out):
        self.out=out;self.entries=[];self.sources={};self.cm_cache={}
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':10,
            'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
            'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'})
        for name in ['01_main_figures','02_supplementary_figures','03_confusion_matrices','04_source_tables','05_manuscript_notes','source']:
            (out/name).mkdir(parents=True,exist_ok=True)

    def source(self,path):
        self.sources[str(path.relative_to(ROOT))]=digest(path);return path

    def frame(self,path):return pd.read_csv(self.source(path))

    def save(self,fig,section,stem,title,caption,placement):
        folder=self.out/section;paths={}
        for fmt in ['png','pdf','svg']:
            target=folder/fmt/(stem+'.'+fmt);target.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(target,dpi=300,bbox_inches='tight',pad_inches=.12)
            paths[fmt]=str(target.relative_to(self.out)).replace('\\','/')
        plt.close(fig)
        self.entries.append(dict(id=stem,title=title,caption=caption,placement=placement,**paths))
        print('FIGURE',stem,flush=True)

    def data(self,frame,name):frame.to_csv(self.out/'04_source_tables'/name,index=False)


def box(ax,x,y,w,h,text,color='#EAF2F8',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=.015',
        linewidth=1,edgecolor='#607080',facecolor=color))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)


def arrow(ax,x1,y1,x2,y2,color='#435466'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,lw=1.2,color=color))


def datasets(pack,config,groups):
    dedup=read(pack.source(ROOT/'dataset/deduplicated/summary.json'));counts=[];inventory=[]
    for d in dedup['datasets']:
        source='c4' if '_c4' in d['source'] else 'c8'
        wave=np.load(pack.source(BD/f'{source}_native_physical_coordinates.npy'))
        for c in d['classes']:counts.append(dict(dataset=source,polymer=c['polymer'],original=c['original'],deduplicated=c['retained'],removed=c['removed']))
        inventory.append(dict(dataset=source,original_records=d['original'],deduplicated_records=d['retained'],removed_exact_duplicates=d['removed'],
            native_coordinates=len(wave),native_min_cm_inverse=wave.min(),native_max_cm_inverse=wave.max(),paired_record_groups=len(groups)))
    counts=pd.DataFrame(counts);pack.data(counts,'dataset_exact_deduplication_counts.csv');pack.data(pd.DataFrame(inventory),'dataset_inventory.csv')
    gc=groups.groupby('label').size().reindex(CLASSES);pack.data(gc.rename_axis('polymer').reset_index(name='paired_record_groups'),'paired_record_group_counts.csv')
    fig=plt.figure(figsize=(13,8.3));gs=fig.add_gridspec(2,1,height_ratios=[1,1.05],hspace=.35)
    ax=fig.add_subplot(gs[0]);ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.01,.65,.20,.23,'C4: 3,000 records\n3,736 native coordinates\n500 / polymer','#E8F1FA')
    box(ax,.01,.12,.20,.23,'C8: 3,000 records\n1,869 native coordinates\n500 / polymer','#FBEDE4')
    box(ax,.29,.65,.20,.23,'Exact deduplication\n2,649 retained\n351 duplicate rows removed','#E8F1FA')
    box(ax,.29,.12,.20,.23,'Exact deduplication\n2,390 retained\n610 duplicate rows removed','#FBEDE4')
    arrow(ax,.215,.765,.28,.765);arrow(ax,.215,.235,.28,.235)
    box(ax,.56,.30,.20,.40,'Union of record IDs\nand duplicate aliases\n\n2,047 record groups\n1 representative /\ngroup / resolution','#E8F5ED')
    arrow(ax,.495,.765,.55,.55);arrow(ax,.495,.235,.55,.45)
    box(ax,.82,.32,.17,.36,'Locked physical grid\n1,868 coordinates\n399.19–4,000.60 cm⁻¹\n\nGroup-safe partitions','#F1ECF8',size=9)
    arrow(ax,.765,.50,.81,.50)
    ax.text(.5,.02,'Conservative record groups; physical specimen independence is unverified.',ha='center',fontsize=9,color='#6A4F32')
    ax.set_title('(a) Record audit and analysis population',loc='left')
    ax=fig.add_subplot(gs[1]);x=np.arange(6);width=.24
    for offset,s,color in [(-width,'c4','#0072B2'),(0,'c8','#E69F00')]:
        y=counts[counts.dataset==s].set_index('polymer').loc[CLASSES,'deduplicated'].to_numpy()
        bars=ax.bar(x+offset,y,width,color=color,label=s.upper()+' exact-deduplicated')
        ax.bar_label(bars,fontsize=8,padding=3)
    bars=ax.bar(x+width,gc.to_numpy(),width,color='#009E73',label='Paired record groups');ax.bar_label(bars,fontsize=8,padding=3)
    ax.axhline(500,color='#777777',ls='--',lw=1,label='Original: 500 per polymer / resolution')
    ax.set_xticks(x);ax.set_xticklabels(CLASSES);ax.set_ylim(0,570);ax.set_ylabel('Number of records / groups')
    ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.12));ax.set_title('(b) Class counts at different audit levels',loc='left',pad=15)
    pack.save(fig,'01_main_figures','Fig01_dataset_audit','Dataset structure and duplicate handling',
        'Original records, within-file exact deduplication and conservative cross-resolution record grouping. All original CSVs were preserved. Group counts are not verified specimen counts. One representative per group and resolution was used in the final protocol. Exact duplicate removal uses numeric equality, without rounding or a similarity threshold.','Materials and methods 2.1, after the dataset description.')
    # Descriptive spectral medians; no new preprocessing or fitted model.
    grid=np.load(pack.source(PC/'common_grid.npy'));signals={}
    for source in ['c4','c8']:
        caches=list((PC/'preprocessing_cache').glob(source+'_*.npz'));assert len(caches)==1
        with np.load(pack.source(caches[0])) as d:signals[source]=d['x'][groups[source+'_index'].to_numpy()]
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True);summary=[]
    for ax,c in zip(axes.flat,CLASSES):
        mask=groups.label.to_numpy()==c
        for s,color in [('c4','#0072B2'),('c8','#D55E00')]:
            q=np.quantile(signals[s][mask],[.1,.5,.9],axis=0)
            ax.plot(grid,q[1],color=color,label=s.upper(),lw=1);ax.fill_between(grid,q[0],q[2],color=color,alpha=.13,lw=0)
            summary.extend(dict(polymer=c,resolution=s,wavenumber=float(w),p10=float(a),median=float(b),p90=float(d)) for w,a,b,d in zip(grid,*q))
        ax.set_title(c+f' (n={int(mask.sum())} groups / resolution)');ax.set_xlim(grid.max(),grid.min());ax.set_ylim(-.08,1.12);ax.grid(alpha=.15)
    axes[0,0].legend();fig.supxlabel('Wavenumber (cm⁻¹)');fig.supylabel('Preprocessed signal (a.u.)');fig.tight_layout()
    pack.data(pd.DataFrame(summary),'spectral_medians_and_p10_p90.csv')
    pack.save(fig,'01_main_figures','Fig02_polymer_spectra','Preprocessed spectra on the common physical grid',
        'Per-polymer median spectra with 10th–90th percentile bands across the paired record representatives, shown for descriptive purposes only. Display summaries do not fit preprocessing or classifiers and use all analysis records. Spectral direction is conventional high-to-low wavenumber. Stored absorbance/transmittance semantics were not verified; the ordinate denotes the numeric preprocessed signal.','Materials and methods 2.2.')
    return counts,inventory


def diagrams(pack,config):
    fig,ax=plt.subplots(figsize=(14,8));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.02,.76,.22,.19,'4 source base classes\nGroup-disjoint train / validation\n2 novel classes excluded','#E8F1FA')
    box(ax,.31,.76,.26,.19,'Initial representation only\nCE-M1 / episodic M1 / PCA\nSelect neural checkpoint on source validation','#E8F1FA',9)
    arrow(ax,.25,.855,.30,.855);box(ax,.65,.76,.30,.19,'FREEZE\nEncoder / PCA and source covariance prior\nNo incremental parameter updates','#E8F5ED',9);arrow(ax,.58,.855,.64,.855)
    box(ax,.02,.43,.29,.20,'Session 0: 4 base classes\n20 references per base class\nReferences drawn from base-train groups','#EDF4FC',9)
    box(ax,.36,.43,.28,.20,'Session 1: register novel class 1\nK = 1 / 5 / 10 / 20 supports\nUpdate only the arriving class state','#FBEDE4',9)
    box(ax,.69,.43,.28,.20,'Session 2: register novel class 2\nK = 1 / 5 / 10 / 20 supports\nKeep previous class states unchanged','#FBEDE4',9)
    arrow(ax,.32,.53,.35,.53);arrow(ax,.65,.53,.68,.53)
    ax.plot([.165,.83],[.69,.69],color='#435466',lw=1)
    ax.plot([.80,.80],[.75,.69],color='#435466',lw=1)
    for x in [.165,.50,.83]:arrow(ax,x,.69,x,.64)
    for x,n in [(.165,4),(.50,5),(.83,6)]:
        arrow(ax,x,.42,x,.33);box(ax,x-.135,.16,.27,.16,f'{n} seen classes × 15 queries\nAccuracy, Macro-F1, per-class recall\nBase / previous / current novel metrics','#F6F6F6',9)
    ax.text(.5,.05,'Three held-out class tasks × three seeds × 200 paired episodes; primary K = 5 / 10.',ha='center',fontsize=11)
    pack.save(fig,'01_main_figures','Fig03_frozen_incremental_workflow','Frozen representation and 4+1+1 protocol',
        'Representation fitting and source-only validation precede class registration. Base reference groups are sampled from legal base-training groups, not from a mutually disjoint extra support split. Novel support/query pools and all evaluation queries remain group-disjoint from representation training/selection. The first novel class is HDPE, PP or PET for the three tasks; the second is LDPE, PS or PVC, respectively.','Methods 2.3–2.6.')
    fig,ax=plt.subplots(figsize=(14,5.3));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.01,.44,.105,.26,'Input\n1,868 × 1','#E8F1FA',10)
    for y,k in [(.73,9),(.44,25),(.15,49)]:
        box(ax,.165,y,.15,.20,f'Conv1D\n16 filters, k={k}\nReLU / same',size=9);arrow(ax,.12,.57,.16,y+.10)
    box(ax,.365,.41,.10,.30,'Concatenate\n48 channels',size=9)
    for y in [.83,.54,.25]:arrow(ax,.32,y,.36,.56)
    box(ax,.51,.41,.15,.30,'[MaxPool ×2\nConv32, k=5]\nrepeated twice\n467 × 32',size=9);arrow(ax,.47,.56,.50,.56)
    box(ax,.705,.41,.12,.30,'Conv8, k=1\n32 ordered bins\nFlatten: 256',size=9);arrow(ax,.665,.56,.70,.56)
    box(ax,.87,.41,.12,.30,'Dense128\nReLU\nEmbedding',color='#E8F5ED',size=9);arrow(ax,.832,.56,.865,.56)
    ax.text(.65,.13,'M1 encoder: 47,400 parameters\nSame architecture for CE and episodic training\nTemporary CE head discarded; no classifier head in the episodic objective',ha='center',fontsize=11)
    pack.save(fig,'02_supplementary_figures','FigS01_M1_architecture','Exact M1 encoder architecture',
        'The locked multiscale M1 definition in encoder_architecture_models.py. The three convolutional branches preserve physical position; ordered spectral-bin pooling precedes the 128-dimensional embedding. CE uses an additional temporary four-class head (516 parameters) only during initial supervised training.','Methods 2.4 and implementation appendix.')
    fig,ax=plt.subplots(figsize=(11.5,4.8));arr=np.array([[0 if config['cases'][c][key]=='c4' else 1 for key in ['encoder','base_references','novel_support','query']] for c in CASES])
    from matplotlib.colors import ListedColormap
    ax.imshow(arr,cmap=ListedColormap(['#D2E5F6','#F9D8BC']),vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(4));ax.set_xticklabels(['Representation fit','Base references','Novel support','Query']);ax.xaxis.tick_top()
    ax.set_yticks(range(6));ax.set_yticklabels(['A: C4 same','B: C8 same','C: C4 → C8','D: C8 → C4','E: target C8 bank','F: target C4 bank'])
    for i in range(6):
        for j in range(4):ax.text(j,i,('C4' if arr[i,j]==0 else 'C8')+('*' if i>=4 and j==1 else ''),ha='center',va='center',fontsize=12)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True);ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='minor',bottom=False,left=False)
    fig.text(.5,.015,'*E/F require 20 labelled target spectra per base class (80 additional target base references).',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1])
    pack.save(fig,'02_supplementary_figures','FigS02_AF_data_access','Six settings and data-access conditions',
        'The representation and covariance prior are source-trained and frozen in every setting. E/F use additional labelled target-resolution base references; gains over C/D are not free or label-free adaptation. Training uses all legal source base-training groups, whereas classifier reference banks use 20 spectra per base class and K per novel class.','Experimental design 3.1; visual companion to Table 1.')


def load_results(pack,config):
    ep=pack.frame(EP/'task_seed_metrics.csv')
    pc=pack.frame(PC/'task_seed_metrics.csv');pc=pc[pc.method.isin(['PCA64_shrinkage','PCA128_shrinkage'])]
    raw=[]
    for task in config['tasks']:
        for seed in [0,1,2]:
            d=pack.frame(BD/task['name']/f'seed_{seed}'/'metrics.csv');d=d[d.method.isin(['raw_1nn','raw_svm'])]
            raw.append(d.groupby(['task','seed','setting','shot','session','method'],as_index=False)[['all_accuracy','all_macro_f1']].mean())
    raw=pd.concat(raw,ignore_index=True)
    # EP table already includes the reused PCA64 reference; retain one authoritative copy.
    ep=ep[ep.method!='PCA64_shrinkage'];units=pd.concat([ep,pc,raw],ignore_index=True)
    keys=['task','seed','setting','shot','session','method'];assert not units.duplicated(keys).any()
    primary=units[(units.session==2)&units.shot.isin([5,10])].groupby(['task','seed','setting','method'],as_index=False)[METRICS].mean()
    pack.data(units,'all_shots_sessions_task_seed.csv');pack.data(primary,'primary_task_seed.csv')
    pack.data(primary.groupby(['setting','method'],as_index=False)[METRICS].mean(),'primary_means.csv')
    return units,primary


def bars(ax,primary,methods,metric,title,ymin=0):
    width=.78/len(methods);x=np.arange(6)
    for j,m in enumerate(methods):
        g=primary[primary.method==m].groupby('setting')[metric].agg(['mean','std']).reindex(CASES)
        pos=x+(j-(len(methods)-1)/2)*width
        ax.bar(pos,100*g['mean'],width*.92,color=COLOR[m],label=LABEL[m],yerr=100*g['std'],
            error_kw={'lw':.7,'capsize':2,'alpha':.6})
    ax.set_xticks(x);ax.set_xticklabels(CASES);ax.set_ylim(ymin,104);ax.set_ylabel(title+' (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)


def result_figures(pack,units,primary):
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    methods=['raw_1nn','raw_svm','PCA128_shrinkage','EP_shrinkage']
    for ax,m,title in zip(axes,['all_accuracy','all_macro_f1'],['Accuracy','Macro-F1']):bars(ax,primary,methods,m,title,55)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center');fig.tight_layout(rect=[0,.10,1,1])
    pack.save(fig,'01_main_figures','Fig04_conventional_baselines','Conventional and class-statistical baselines',
        'Final-session 5/10-shot means for spectrum 1-NN/SVM, frozen PCA128 and EP-M1 with shrinkage. Error bars are descriptive SD across nine task–seed means, not standard errors or confidence intervals. The seeds/tasks share data; repeated episodes are not independent specimens. The ordinate starts at 55%. SVM is refitted on the permitted retained reference bank.','Results 4.1, following Table 2.')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5));methods=['PCA64_shrinkage','PCA128_shrinkage','CE_shrinkage','EP_shrinkage']
    for ax,m,title in zip(axes,['all_accuracy','all_macro_f1'],['Accuracy','Macro-F1']):bars(ax,primary,methods,m,title,60)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center');fig.tight_layout(rect=[0,.10,1,1])
    pack.save(fig,'01_main_figures','Fig05_frozen_representations','Representations under the fixed shrinkage head',
        'PCA64/PCA128, CE-M1-128 and EP-M1-128 with the same numerical shrinkage rule. Final-session primary 5/10-shot results; error bars are descriptive SD over nine dependent task–seed means. Axes start at 60%. Equal dimensions or a common numeric ridge do not equalise feature scales or relative regularisation.','Results 4.2 / Table 3; PCA64 control also supports Results 4.6.')
    paired=pack.frame(EP/'paired_differences.csv');pack.data(paired,'episodic_vs_CE_and_head_paired_differences.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5));headcolors=['#CC79A7','#8064A2','#D55E00']
    for ax,metric,title in zip(axes,['all_accuracy_delta_pp','seen_novel_accuracy_delta_pp'],['Overall accuracy change','Combined novel-class accuracy change']):
        for j,(head,color) in enumerate(zip(['euclidean','full','shrinkage'],headcolors)):
            g=paired[paired.comparison=='episodic_vs_CE_'+head].groupby('setting')[metric].agg(['mean','std']).reindex(CASES)
            ax.errorbar(np.arange(6)+(j-1)*.18,g['mean'],yerr=g['std'],fmt='o',color=color,capsize=3,label=head.capitalize())
        ax.axhline(0,color='#444444',lw=1);ax.set_xticks(range(6));ax.set_xticklabels(CASES);ax.set_ylabel('EP minus CE (percentage points)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(title='Fixed inference head');fig.tight_layout()
    pack.save(fig,'01_main_figures','Fig06_episodic_training_ablation','Episodic versus CE training at a fixed distance head',
        'Paired differences in final-session overall and combined novel-class accuracy. Each unit averages 200 paired repeats and the primary 5/10 shots. Points show the mean, error bars descriptive SD across nine dependent task–seed units. The training contrast also changes sampling roles and validation objectives; it is not an isolated pure-loss causal experiment.','Results 4.2, after the negative finding about Euclidean novel recognition.')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5));methods=['EP_euclidean','EP_full','EP_shrinkage']
    for ax,m,title in zip(axes,['all_accuracy','seen_novel_accuracy'],['Overall accuracy','Combined novel-class accuracy']):bars(ax,primary,methods,m,title,0)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center');fig.tight_layout(rect=[0,.10,1,1])
    pack.save(fig,'01_main_figures','Fig07_EP_distance_head_ablation','Distance-head ablation on the same frozen EP encoder',
        'Only the inference head changes: squared Euclidean, independent class full covariance with ridge .001, or class-specific shrinkage with tau5 and the fixed legal source prior. Overall accuracy and novel accuracy can move in opposite directions, notably in C/D. Error bars show descriptive task–seed SD.','Results 4.3 / Table 4.')
    fig,axes=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    methods=['EP_euclidean','EP_full','EP_shrinkage','PCA64_shrinkage','PCA128_shrinkage']
    for ax,case in zip(axes.flat,CASES):
        for method in methods:
            g=units[(units.setting==case)&(units.method==method)&(units.session==2)].groupby('shot').all_accuracy.mean()
            ax.plot(g.index,100*g.values,'o-',lw=1.4,color=COLOR[method],label=LABEL[method],ms=4)
        ax.set_title(case);ax.set_xticks([1,5,10,20]);ax.set_ylim(50,101);ax.grid(alpha=.2)
    fig.supxlabel('Support spectra per arriving novel class (K)');fig.supylabel('Final-session accuracy (%)')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.02));fig.tight_layout(rect=[.02,.07,1,1])
    pack.save(fig,'01_main_figures','Fig08_shot_sensitivity','Recognition versus novel support size',
        'All prespecified novel support sizes, 1/5/10/20, using nested support prefixes and the same query draws. Curves are means over task–seed units; their connecting lines aid reading and are not fitted performance models. No support size was selected after inspection.','Results 4.3–4.4; use as supplementary if main text space is limited.')
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    trajectory=units[(units.method=='EP_shrinkage')&units.shot.isin([5,10])].groupby(['task','seed','setting','session'],as_index=False)[METRICS].mean()
    pack.data(trajectory,'EP_shrinkage_incremental_trajectories.csv')
    for ax,case in zip(axes.flat,CASES):
        g=trajectory[trajectory.setting==case].groupby('session')[METRICS].mean()
        for field,color,label,style in [('all_accuracy','#333333','All seen','o-'),('base_accuracy','#0072B2','Base classes','s-'),('seen_novel_accuracy','#D55E00','Combined novel','^-')]:
            ax.plot(g.index,100*g[field],style,color=color,label=label,lw=1.5)
        first=[100*g.loc[1,'current_novel_accuracy'],100*g.loc[2,'previous_novel_accuracy']]
        ax.plot([1,2],first,'D--',color='#8064A2',label='First novel only',lw=1.2)
        ax.set_title(case);ax.set_xticks([0,1,2]);ax.set_xticklabels(['0: four base','1: + novel 1','2: + novel 2'],rotation=15,fontsize=8);ax.set_ylim(55,103);ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center');fig.supylabel('Accuracy (%)');fig.tight_layout(rect=[.02,.07,1,1])
    pack.save(fig,'01_main_figures','Fig09_incremental_retention','Base and novel recognition across incremental sessions',
        'EP-M1 with shrinkage, primary 5/10-shot averages. The dashed first-novel curve follows the same first introduced class from session1 to session2, even though its stored state is unchanged. The combined novel curve averages one class in session1 and two classes in session2. No novel accuracy is defined in session0. Curves describe decision competition rather than encoder-update forgetting.','Results 4.4 / Table 5.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));methods=['CE_shrinkage','EP_shrinkage','PCA128_shrinkage'];alignment=[]
    for ax,(before,after,title) in zip(axes,[('C','E','C4-trained → C8 queries'),('D','F','C8-trained → C4 queries')]):
        for j,m in enumerate(methods):
            d=primary[(primary.method==m)&primary.setting.isin([before,after])].pivot(index=['task','seed'],columns='setting',values='all_accuracy')*100
            xs=np.array([0,1])+(j-1)*.035
            for _,row in d.iterrows():ax.plot(xs,row[[before,after]],color=COLOR[m],alpha=.16,lw=.7)
            mean=d[[before,after]].mean();ax.plot(xs,mean,'o-',color=COLOR[m],label=LABEL[m],lw=2.2)
            for x,y in zip(xs,mean):
                if x>.5:
                    dy=[-1.0,.6,1.0][j]
                    ax.annotate(f'{y:.2f}',xy=(x,y),xytext=(x+.12,y+dy),ha='left',va='center',fontsize=8,color=COLOR[m],
                        arrowprops=dict(arrowstyle='-',lw=.6,color=COLOR[m]))
                else:ax.text(x-.025,y,f'{y:.2f}',ha='right',va='center',fontsize=8,color=COLOR[m])
            alignment.extend(dict(task=i[0],seed=i[1],direction=before+'→'+after,method=m,gain_pp=row[after]-row[before]) for i,row in d.iterrows())
        ax.set_xlim(-.4,1.4);ax.set_ylim(60,102);ax.set_xticks([0,1]);ax.set_xticklabels([before+': source base bank',after+': target base bank']);ax.set_ylabel('Overall accuracy (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.text(.5,.11,'Target alignment uses 20 extra labelled target references per base class; encoder / PCA and source prior stay frozen.',ha='center',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center');fig.tight_layout(rect=[0,.18,1,1]);pack.data(pd.DataFrame(alignment),'reference_alignment_paired_gains.csv')
    pack.save(fig,'01_main_figures','Fig10_reference_alignment','Effect of target-resolution base references',
        'Paired source-bank to target-bank transitions C→E and D→F. Thin lines show nine task–seed pairs and bold lines their means. Representation parameters and the source covariance prior are fixed. Target-aligned banks require 80 labelled target base references, a different data-access condition from C/D.','Results 4.5 / Table 6.')


def get_cm(pack,root,task,seed,case,shot,method,new=False):
    folder=root/task/f'seed_{seed}';prefix='new_' if new else ''
    key=str(folder)+prefix
    if key not in pack.cm_cache:
        idx=read(pack.source(folder/(prefix+'confusion_index.json')));arr=np.load(pack.source(folder/(prefix+'confusions.npz')))
        lookup={(m['setting'],m['shot'],m['session'],m['method']):(m['classes'],arr[token]) for token,m in idx.items()}
        pack.cm_cache[key]=lookup
    return pack.cm_cache[key][case,shot,2,method]


def cm_plot(ax,cm,classes,novel,title,large=False):
    norm=100*cm/cm.sum(axis=1,keepdims=True);ax.imshow(norm,cmap='Blues',vmin=0,vmax=100)
    ticks=[c+('*' if c in novel else '') for c in classes]
    ax.set_xticks(range(len(classes)));ax.set_xticklabels(ticks,rotation=65,ha='right',fontsize=8 if large else 7)
    ax.set_yticks(range(len(classes)));ax.set_yticklabels(ticks,fontsize=8 if large else 7);ax.set_title(title,fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):ax.text(j,i,f'{norm[i,j]:.1f}',ha='center',va='center',fontsize=7.2 if large else 6.3,color='white' if norm[i,j]>55 else '#222222')
    ax.set_xlabel('Predicted',fontsize=8);ax.set_ylabel('True',fontsize=8)


def confusion_figures(pack,config):
    refs={'PCA128_shrinkage':(PC,'PCA128_shrinkage',False),'PCA64_shrinkage':(BD,'PCA64_shrinkage',False),
          'CE_shrinkage':(BD,'M1_shrinkage',False),'EP_shrinkage':(EP,'EP_shrinkage',True),
          'EP_full':(EP,'EP_full',True),'EP_euclidean':(EP,'EP_euclidean',True)}
    pooled={};records=[]
    for task in config['tasks']:
        name=task['name'];novel=[s[0] for s in task['novel_sessions']]
        for method,(root,oldname,new) in refs.items():
            for shot in [1,5,10,20]:
                for case in CASES:
                    arrays=[get_cm(pack,root,name,seed,case,shot,oldname,new) for seed in [0,1,2]]
                    classes=arrays[0][0];assert all(cs==classes for cs,_ in arrays)
                    cm=sum(a for _,a in arrays);assert np.all(cm.sum(1)==200*15*3)
                    pooled[name,method,shot,case]=(classes,cm)
                    for i,c in enumerate(classes):
                        for j,pred in enumerate(classes):records.append(dict(task=name,method=method,shot=shot,setting=case,true=c,predicted=pred,count=int(cm[i,j]),row_percentage=100*cm[i,j]/cm[i].sum()))
                if shot not in [5,10]:continue
                fig,axes=plt.subplots(2,3,figsize=(12,8))
                for ax,case in zip(axes.flat,CASES):
                    cs,cm=pooled[name,method,shot,case];cm_plot(ax,cm,cs,novel,case)
                fig.suptitle(TASKLABEL[name]+f' | {LABEL[method]} | {shot}-shot | final session',fontsize=12)
                fig.text(.5,.01,'Rows sum to 100%; * marks genuine novel classes. Pooled repeated predictions are not independent specimens.',ha='center',fontsize=8)
                fig.tight_layout(rect=[0,.04,1,.96])
                section='03_confusion_matrices/'+name+'/'+str(shot)+'shot'
                pack.save(fig,section,method+'_'+str(shot)+'shot',TASKLABEL[name]+': '+LABEL[method]+f', {shot}-shot',
                    f'Final six-class row-normalised confusion percentages for all A–F settings. Counts pool 200 episodes and three seeds within {name}; each row has 9,000 query occurrences, not 9,000 unique specimens. Asterisks mark the two genuine novel classes. Exact count and percentage tables for all four shots are in confusion_counts_all_shots.csv.','Detailed confusion supplement for Results 4.3–4.5.')
    records=pd.DataFrame(records);pack.data(records,'confusion_counts_all_shots.csv')
    main_confusion_pages(pack,config,pooled)
    # Key confusing-polymer summaries, without mixing base and novel roles.
    key=[]
    for task,a,b in [('holdout_hdpe_ldpe','HDPE','LDPE'),('holdout_pp_ps','PP','PS')]:
        d=records[(records.task==task)&records.shot.isin([5,10])]
        for case in CASES:
            for method in refs:
                z=d[(d.setting==case)&(d.method==method)]
                value=z[((z.true==a)&(z.predicted==b))|((z.true==b)&(z.predicted==a))].row_percentage.mean()
                key.append(dict(task=task,setting=case,method=method,pair=a+'/'+b,mean_bidirectional_error_percentage=value))
    pack.data(pd.DataFrame(key),'genuine_novel_pair_confusions.csv')


def main_confusion_pages(pack,config,pooled):
    # Six panels at a paper-friendly width; A/B/E/F remain in the complete CM supplement.
    for n,name in enumerate(['holdout_hdpe_ldpe','holdout_pp_ps'],11):
        task=next(t for t in config['tasks'] if t['name']==name);novel=[s[0] for s in task['novel_sessions']]
        fig,axes=plt.subplots(2,3,figsize=(8.4,6.8))
        for i,case in enumerate('CD'):
            for j,method in enumerate(['EP_full','EP_shrinkage','PCA128_shrinkage']):
                cs,cm=pooled[name,method,10,case];cm_plot(axes[i,j],cm,cs,novel,case+' | '+LABEL[method],large=True)
        fig.suptitle(TASKLABEL[name]+' | 10-shot final session',fontsize=11)
        fig.text(.5,.01,'Row percentages; * genuine novel class. C: C4 → C8; D: C8 → C4.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.96])
        pack.save(fig,'01_main_figures',f'Fig{n:02d}_CM_'+name,TASKLABEL[name]+': cross-resolution confusion',
            'Fixed 10-shot final session, three-seed pooled confusion, shown without selecting a best seed. Rows are true classes and columns predictions; each row sums to 100%. C/D retain source base references. Compare EP full/shrinkage and the dimension-matched PCA128 shrinkage reference. Asterisks mark genuine held-out classes. All six settings including target-aligned E/F are provided in the complete confusion supplement.','Results 4.3–4.4; one page may move to supplementary material.')


def supplements(pack,config,groups):
    rows=[]
    for task in config['tasks']:
        for seed in [0,1,2]:
            path=BD/task['name']/f'seed_{seed}'/'split.json';split=read(pack.source(path));table=groups.copy();table['role']=split['roles']
            for (c,role),n in table.groupby(['label','role']).size().items():rows.append(dict(task=task['name'],seed=seed,polymer=c,role=role,record_groups=n))
    rows=pd.DataFrame(rows);pack.data(rows,'group_safe_split_counts.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    roles=['base_train','base_validation','base_query','novel_support_pool','novel_query_pool']
    colors=['#0072B2','#56B4E9','#B7CADD','#E69F00','#F6D792']
    for ax,task in zip(axes,config['tasks']):
        d=rows[(rows.task==task['name'])&(rows.seed==0)].pivot(index='polymer',columns='role',values='record_groups').reindex(CLASSES).fillna(0);bottom=np.zeros(6)
        for role,color in zip(roles,colors):
            y=d[role].to_numpy() if role in d else np.zeros(6);ax.bar(np.arange(6),y,bottom=bottom,color=color,label=role.replace('_',' '));bottom+=y
        ax.set_xticks(range(6));ax.set_xticklabels(CLASSES,rotation=45);ax.set_title(TASKLABEL[task['name']],fontsize=10);ax.set_ylabel('Record groups')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',fontsize=8);fig.tight_layout(rect=[0,.10,1,1])
    pack.save(fig,'02_supplementary_figures','FigS03_group_safe_partitions','Role counts under the three held-out class tasks',
        'Counts shown for seed0; class-wise role counts are identical across seeds although group membership changes. Base reference spectra are selected from base_train, so they are not an additional mutually disjoint split. Novel support/query pools remain separate; only K support groups and 15 query groups per seen class enter a given episode. C4/C8 counterparts inherit the same group role.','Methods 2.1/2.3 and data-audit supplement.')
    geom=pack.frame(EP/'feature_geometry.csv');pack.data(geom,'CE_EP_feature_geometry.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    for ax,metric,title in zip(axes,['mean_training_norm','effective_rank','ridge_over_mean_variance'],['Mean training feature norm','Prior covariance effective rank','Ridge / mean residual variance']):
        for j,obj in enumerate(['CE','EP']):
            g=geom[geom.objective==obj].groupby('source')[metric].mean().reindex(['c4','c8'])
            ax.bar(np.arange(2)+(j-.5)*.3,g,.28,color=['#0072B2','#D55E00'][j],label=obj)
        ax.set_xticks([0,1]);ax.set_xticklabels(['C4','C8']);ax.set_title(title,fontsize=10)
        if metric=='ridge_over_mean_variance':ax.set_yscale('log')
    axes[0].legend();fig.tight_layout()
    pack.save(fig,'02_supplementary_figures','FigS04_feature_geometry','Feature scale and effective regularisation',
        'Source base-training embeddings only. Mean feature norm, effective rank of the pooled residual covariance and numeric ridge divided by mean residual variance. The last panel uses a logarithmic ordinate. A fixed ridge value does not impose equal relative regularisation on CE/EP representations; effective rank alone does not establish better novel-class recognition.','Discussion 5.2–5.3.')
    costs=pack.frame(EP/'training_comparison.csv');pca=pack.frame(PC/'pca_fit_costs.csv')
    heads=pack.frame(PC/'costs.csv');old=pack.frame(BD/'costs.csv')
    r=[]
    for obj,g in costs.groupby('objective'):r.append(dict(method=obj+' M1 initial training',mean_seconds=g.training_seconds.mean(),scope='includes validation/checkpoint I/O; recorded CPU runs'))
    r.append(dict(method='PCA128 initial fit',mean_seconds=pca.fit_seconds.mean(),scope='PCA fit only; no validation or neural training'))
    pack.data(pd.DataFrame(r),'representation_fit_costs.csv')
    h=heads[(heads.session>0)&heads.shot.isin([5,10])].head_local_registration_ms.median()
    svm=old[(old.method=='raw_svm')&(old.session>0)&old.shot.isin([5,10])].full_assembly_refit_ms.median()
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    axes[0].bar(['CE training','EP training','PCA128 fit'],[r[0]['mean_seconds'],r[1]['mean_seconds'],r[2]['mean_seconds']],color=['#0072B2','#D55E00','#009E73']);axes[0].set_yscale('log');axes[0].set_ylabel('Mean recorded seconds (log scale)');axes[0].set_title('Initial representation cost')
    axes[1].bar(['PCA128 new-class\nregistration','Spectrum SVM\nfull-bank refit'],[h,svm],color=['#009E73','#444444']);axes[1].set_ylabel('Median recorded milliseconds');axes[1].set_title('Incremental head update, 5/10 shots')
    fig.tight_layout()
    pack.save(fig,'02_supplementary_figures','FigS05_recorded_compute_costs','Recorded representation and class-update costs',
        'Descriptive timings from the archived runs, not a controlled speed benchmark. Initial neural training includes validation/checkpoint work, whereas PCA time covers fitting only. Incremental head times exclude encoding/preprocessing: PCA registers only the arriving class; SVM refits the entire retained labelled bank. Different runs and operation scopes prohibit interpreting the ratios as intrinsic algorithmic speedups.','Results 4.6; supplementary cost context only.')


def database_schema(pack):
    fig,ax=plt.subplots(figsize=(14,8));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.02,.61,.28,.31,'Spectral CSV (one file / resolution)\n\nRow 1: generic column indices\nRow 2: physical wavenumber metadata\nRows 3+: ID | polymer | class ID | values\nC4: 3,736 values; C8: 1,869 values','#E8F1FA',9)
    box(ax,.37,.61,.26,.31,'Duplicate row mapping CSV\n\nsource_csv_row / source_id\npolymer / class_id / status\nretained_source_csv_row\nretained_source_id','#FBEDE4',9)
    arrow(ax,.31,.765,.36,.765)
    box(ax,.70,.61,.28,.31,'Paired group representatives CSV\n\ngroup / label\nc4_index / c8_index\nc4_id / c8_id / alias counts\n2,047 conservative record groups','#E8F5ED',9)
    arrow(ax,.64,.765,.69,.765)
    box(ax,.02,.15,.28,.27,'common_grid.npy\n\n1,868 explicit physical coordinates\n399.1927–4,000.6045 cm⁻¹\nShared for C4 / C8','#F1ECF8',9)
    box(ax,.37,.15,.26,.27,'split.json (task / seed)\n\nbase train / validation / query\nnovel support / query pools\nRoles shared across C4 / C8','#EDF4FC',9)
    box(ax,.70,.15,.28,.27,'episodes.json (task / seed)\n\n20 base references / class\nNested novel support order (K≤20)\n15 query groups / seen class\n200 matched evaluation episodes','#EDF4FC',9)
    arrow(ax,.83,.60,.50,.43);arrow(ax,.64,.285,.69,.285)
    ax.text(.5,.035,'IDs, CSV row numbers and loaded-array indices are distinct; record groups are not verified specimen IDs.',ha='center',fontsize=10)
    pack.save(fig,'02_supplementary_figures','FigS06_database_file_schema','Database file schema and evaluation manifests',
        'The actual CSV layout includes a separate physical-wavenumber metadata row. Duplicate mapping relates original rows to retained records; paired group indices address the loaded deduplicated arrays. Split/episode manifests express legal roles and matched draws. A CSV row index must not be treated as a cross-resolution specimen identity.','Data-availability and reproducibility appendix; companion to Fig01.')


def audit_manuscript(pack,docx,primary):
    root=ET.fromstring(zipfile.ZipFile(pack.source(docx)).read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    texts=[''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in root.findall('.//w:p',ns)]
    (pack.out/'05_manuscript_notes/manuscript_text_snapshot.txt').write_text('\n'.join(f'{i}: {s}' for i,s in enumerate(texts)),encoding='utf-8')
    tables=[[[''.join(t.text or '' for t in c.findall('.//w:t',ns)) for c in row.findall('w:tc',ns)] for row in tb.findall('w:tr',ns)] for tb in root.findall('.//w:tbl',ns)]
    mean=primary.groupby(['setting','method'])[METRICS].mean();issues=[]
    def check(t,row,col,expected):
        content=tables[t-1][row][col];numbers=re.findall(r'[-+]?\d+(?:\.\d+)?',content.replace('−','-'))
        for actual,desired in zip(numbers,np.atleast_1d(expected)):
            if abs(float(actual)-float(desired))>.00500001:issues.append(dict(table=t,row=row,column=col,manuscript=content,expected_2dp=' / '.join(f'{v:.2f}' for v in np.atleast_1d(expected))))
    for r,c in enumerate(CASES,1):
        for col,m in [(1,'raw_1nn'),(2,'raw_svm')]:check(2,r,col,100*mean.loc[(c,m),'all_accuracy'])
        for col,m in [(1,'PCA128_shrinkage'),(2,'CE_shrinkage'),(3,'EP_shrinkage')]:check(3,r,col,100*mean.loc[(c,m),['all_accuracy','all_macro_f1']].to_numpy())
        for col,m in [(1,'EP_euclidean'),(2,'EP_full'),(3,'EP_shrinkage')]:check(4,r,col,100*mean.loc[(c,m),'all_accuracy'])
        check(4,r,4,100*(mean.loc[(c,'EP_shrinkage'),'all_accuracy']-mean.loc[(c,'EP_full'),'all_accuracy']))
    for r,c in enumerate('ABCD',1):
        for col,m in [(1,'base_accuracy'),(2,'seen_novel_accuracy'),(3,'base_to_novel_error_rate')]:check(5,r,col,100*mean.loc[(c,'EP_shrinkage'),m])
        check(5,r,4,mean.loc[(c,'EP_shrinkage'),'previous_novel_change_pp'])
    pack.data(pd.DataFrame(issues,columns=['table','row','column','manuscript','expected_2dp']),'manuscript_numeric_rounding_audit.csv')
    write(pack.out/'05_manuscript_notes/manuscript_audit.json',dict(source_sha256=digest(docx),paragraphs=len(texts),tables=len(tables),embedded_figures=len([n for n in zipfile.ZipFile(docx).namelist() if n.startswith('word/media/')]),numeric_issues=issues))
    review='''# 稿件与图表检查

已读取 FTIR_Manuscript.docx 的全部段落和6张表。目前文件中没有嵌入图片。稿件已经保留 PCA 优于网络、episodic 欧氏新类识别未改善、收缩在 C/D 中存在基类取舍等关键负结果；不需要为了叙事再增加实验。

## 优先补充或修正

1. **数据去重有两个层次。** C4/C8 原始各3,000条；按单库精确数值相等去重后分别2,649/2,390条，移除重复行351/610条。原始数据库未删除。随后跨分辨率共享记录ID和两库重复别名传递合并，形成2,047个保守记录组；每组每个分辨率选一个代表参与最终实验。不能把2,047称为已验证的独立实物数量。Fig01与数据计数CSV可直接支撑这一段。
2. **2.1 的分组描述需更精确。** 目前“training, validation, support and query were separated”容易让读者认为所有支持集都与训练集独立。实际上20条/类的 base reference 来自 base_train；新类的 support/query 池互斥，query 与 train/validation 均隔离。图中已明确这一点。
3. **2.2 漏写第一步变换。** 实际顺序为原生轴排序 → max(y)−y → 原生轴airPLS基线扣除 → 原生轴max-abs归一化 → cubic插值到固定1,868坐标。airPLS默认 lambda=100、porder=1、itermax=15。请避免把未知单位直接标为吸光度；光谱图已采用 preprocessed signal (a.u.)。
4. **2.4 可直接采用附图中的精确结构。** M1为三分支Conv1D(16,k=9/25/49)，两次[MaxPool2 + Conv32,k=5]，Conv8,k=1，32个有序位置bin，Flatten256，Dense128/ReLU；编码器47,400参数。CE临时四分类头额外516参数，评估时丢弃。
5. **同域/迁移与数据访问须分清。** E/F额外需要每基类20条目标分辨率参考，即80条目标基类标签，不是无标签域适应。PCA/M1拟合可访问合法的全部base-train，而原谱1-NN/SVM分类库仅使用每基类20条和每新类K条。分类参考配对不等于所有算法的初始训练数据访问完全相同。
6. **图中的误差棒不是独立样本置信区间。** 它们是9个相关task/seed均值的描述性SD；不能把200次episode当200个独立实物。CM展示池化重复预测百分比，不应把计数解释成新的样本量。
7. **成本措辞。** 0.279/0.277 ms应称 batch64 下按光谱分摊的推断时间，且是每模型中位数再求均值，不是单条输入的独立调用延迟。训练、PCA拟合与局部注册计时范围不同，只作描述性对照。
8. **投稿前仍需核对原始来源。** 吸光度/透过率语义、原始采集仪器与样本独立性、作者信息和参考文献未在此次制图中外部核验。不能仅凭画图补成已验证事实。原稿末尾 author verification notes 需在完成核对后删除。

## 数值检查

表2–5已从源CSV按未四舍五入的task/seed均值复核；详见 manuscript_numeric_rounding_audit.csv。'''
    if issues:review+='\n\n'+ '\n'.join(f'- 表{i["table"]}，第{i["row"]}个数据行，第{i["column"]}列：{i["manuscript"]} 应为 **{i["expected_2dp"]}**。' for i in issues)
    else:review+=' 未发现超出两位小数舍入容差的差异。'
    review+='''

## 可替换的数据段落（英文）

Within each resolution subset, exact numeric duplicate spectra were removed without rounding or a similarity threshold, retaining 2,649 C4 and 2,390 C8 records from 3,000 records each. The original files were preserved. Shared original record identifiers and both duplicate-alias maps were then unioned conservatively across resolutions, yielding 2,047 record components. One representative per component and resolution was used. These components provide record-level blocking, not verified physical specimen identities. Base training, validation and query groups were disjoint; the 20 base reference spectra per class were sampled from the legal base-training groups. Held-out novel classes used separate support and query pools and were excluded from representation fitting and source validation.

## 图表选用建议

正文推荐Fig01、03、05、06、07、09、10，再选Fig11或Fig12的一张核心混淆图；Fig02光谱、Fig04传统基线、Fig08全shot曲线可按期刊篇幅移入补充。FigS01–S06与按任务整理的完整CM作为补充材料。不要把同一结果的大表和全部对应图都重复塞入正文。图号只是本文件夹的稳定编号，正式投稿时可按最终选图重新连续编号。
'''
    (pack.out/'05_manuscript_notes/Manuscript_review_ZH.md').write_text(review,encoding='utf-8')


def finish(pack,counts,inventory,docx,groups,config):
    pack.entries.sort(key=lambda e:e['png'])
    structure='''# 数据库文件结构与处理关系

原始文件保持不变；“删除重复”指生成新的精确去重副本，不是删除原始文件。

```
dataset/
  FTIR_PLastics500_c4.csv                  原始3,000条
  FTIR_PLastics500_c8.csv                  原始3,000条
  deduplicated/
    FTIR_PLastics500_c4_deduplicated.csv   2,649条
    FTIR_PLastics500_c8_deduplicated.csv   2,390条
    FTIR_PLastics500_c4_row_mapping.csv   原始行→保留行/ID的映射
    FTIR_PLastics500_c8_row_mapping.csv
    summary.json                         原始计数、逐类去重数、源文件哈希
results/bidirectional_c4_c8_final/
  paired_group_representatives.csv        2,047组，各组C4/C8代表行索引与ID
  all_original_ID_groups.csv              全部原始ID→保守记录组
  common_grid.npy                        1,868个固定物理波数坐标
  c4_native_physical_coordinates.npy      3,736个原生坐标
  c8_native_physical_coordinates.npy      1,869个原生坐标
  <held-out-task>/seed_<0|1|2>/
    split.json                           base train/validation/query，新类support/query池
    episodes.json                        200次配对支持/查询索引，1/5/10/20为同一支持顺序的前缀
    sample_manifest.csv                  记录组及其角色
```

## CSV 的真实布局

光谱CSV第1行为列索引header，第2行为波数元数据；从第3行开始才是光谱记录。每条记录的第0列为source ID，第1列为聚合物名称，第2列为class ID，第3列起为光谱值；波数应读取第2行的第3列起，不能把列索引header当波数。C4有3,736个原生光谱值，C8有1,869个。

row_mapping.csv字段：source_csv_row, source_id, polymer, class_id, status, retained_source_csv_row, retained_source_id。

paired_group_representatives.csv中的c4_index/c8_index指向去重载入后的行数组，不是原始CSV行号；source ID与CSV行号也不是同一概念。禁止按去重后行号猜测跨分辨率同一实物。

## 当前计数

| 聚合物 | 原始每库 | C4精确去重 | C8精确去重 | 最终配对记录组 |
| --- | ---: | ---: | ---: | ---: |
'''
    for c in CLASSES:
        a=counts[(counts.dataset=='c4')&(counts.polymer==c)].deduplicated.iloc[0];b=counts[(counts.dataset=='c8')&(counts.polymer==c)].deduplicated.iloc[0]
        structure+=f'| {c} | 500 | {a} | {b} | {int((groups.label==c).sum())} |\n'
    structure+='\n谱值单位尚未源文档确认；图示使用预处理数值单位。去重依据为精确数值相等；保守配对组依据共享原始ID、重复别名与最终共同网格精确指纹，无物理实物独立性保证。\n'
    (pack.out/'05_manuscript_notes/Database_structure_ZH.md').write_text(structure,encoding='utf-8')
    captions='# Figure captions and manuscript placement\n\nAll figures use existing data/results only. No encoder, PCA or classifier was fitted during figure preparation.\n\n'
    for e in pack.entries:captions+=f'## {e["id"]}: {e["title"]}\n\n{e["caption"]}\n\nSuggested placement: {e["placement"]}\n\n'
    (pack.out/'05_manuscript_notes/Figure_captions_EN.md').write_text(captions,encoding='utf-8')
    catalog=['<!doctype html><html><head><meta charset="utf-8"><title>FTIR manuscript figure catalogue</title><style>body{font-family:Arial,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#253443}img{max-width:100%;height:auto}article{border-bottom:1px solid #ddd;padding:20px 0}a{margin-right:15px}p{line-height:1.5}nav{background:#edf4fa;padding:15px}</style></head><body>',
        '<h1>FTIR 论文图表目录</h1><p>PNG 300 dpi，PDF/SVG可缩放。源数据图表均为已有实验汇总；无新训练。* 标记 genuine novel 类。</p>',
        '<nav><a href="05_manuscript_notes/Manuscript_review_ZH.md">稿件检查</a><a href="05_manuscript_notes/Database_structure_ZH.md">数据库结构</a><a href="05_manuscript_notes/Figure_captions_EN.md">英文图注</a></nav>']
    for e in pack.entries:
        catalog.append('<article><h2>'+html.escape(e['id']+' — '+e['title'])+'</h2><p>'+html.escape(e['placement'])+'</p><p>'+''.join('<a href="'+e[fmt]+'">'+fmt.upper()+'</a>' for fmt in ['png','pdf','svg'])+'</p><img loading="lazy" src="'+e['png']+'"><p>'+html.escape(e['caption'])+'</p></article>')
    catalog.append('</body></html>');(pack.out/'figure_catalog.html').write_text('\n'.join(catalog),encoding='utf-8')
    pd.DataFrame(pack.entries).to_csv(pack.out/'figure_index.csv',index=False)
    notes='''# FTIR 论文图表包

先打开 figure_catalog.html 浏览所有图，再按英文图注和插入建议选图。原始 FTIR_Manuscript.docx 未修改。

- 01_main_figures：12张候选正文图；png/pdf/svg分格式整理。
- 02_supplementary_figures：网络结构、A–F数据访问、分组角色、特征几何、描述性成本和数据库文件结构，共6张。
- 03_confusion_matrices：三种留出任务×5/10-shot×六种方法，共36页；每页包含A–F六张CM。所有1/5/10/20-shot的CM计数/百分比另有CSV。
- 04_source_tables：每张统计图对应的task/seed或更细级汇总、去重与数据库计数、混淆计数和稿件数值核查。
- 05_manuscript_notes：中文稿件检查、数据库结构、英文图注、原稿纯文本快照。
- source：可复现制图脚本与论文图表manifest。

图仅由现有数据与保存结果生成，没有重训模型、重拟合PCA或新增分类实验。图中误差棒为相关task/seed均值的描述性SD，不是独立样本置信区间。CM是重复预测的池化计数；每个真类9,000次查询出现不等于9,000个独立实物。PCA128与M1均128维，但尺度和相对正则强度不同。

复现（在项目根目录）：

```powershell
& 'C:\\Users\\31734\\anaconda3\\envs\\fewshotlearning\\python.exe' -X utf8 build_ftir_manuscript_figures_v1.py --output 'paper draft/FTIR_Manuscript_Figures_v1'
```

原始数据、旧实验结果和原稿均被只读访问；输入哈希记入figure_manifest.json。全部图可用于排版；具体主文/补充取舍见稿件检查。
'''
    (pack.out/'README_ZH.md').write_text(notes,encoding='utf-8')
    for name,h in pack.sources.items():assert digest(ROOT/name)==h
    script=Path(__file__);shutil.copy2(script,pack.out/'source'/script.name)
    write(pack.out/'figure_manifest.json',dict(manuscript=str(docx),manuscript_sha256=digest(docx),plot_generator_sha256=digest(script),
        main_figures=12,supplementary_figures=6,confusion_pages=36,total_pages=len(pack.entries),formats=['300-dpi PNG','PDF','SVG'],
        source_hashes=pack.sources,new_training_runs=0,new_PCA_fits=0,old_sources_unchanged=True,confusion_percentages='row-normalised; pooled repeated predictions, not independent specimens',
        aggregation='mean repeats and 5/10 shots within task/seed; then mean across nine dependent task/seed units; descriptive SD only'))
    assert len(pack.entries)==54
    print('COMPLETE:',pack.out,flush=True)


def run(out,docx):
    out=out.resolve();out.mkdir(parents=True,exist_ok=True);pack=Pack(out)
    config=read(pack.source(BD/'experiment_manifest.json'));groups=pack.frame(BD/'paired_group_representatives.csv')
    for p in [BD,EP,PC]:assert read(pack.source(p/'protocol_audit.json'))['status']=='passed'
    counts,inventory=datasets(pack,config,groups);diagrams(pack,config)
    units,primary=load_results(pack,config);result_figures(pack,units,primary)
    confusion_figures(pack,config);supplements(pack,config,groups);database_schema(pack);audit_manuscript(pack,docx,primary)
    finish(pack,counts,inventory,docx,groups,config)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'paper draft/FTIR_Manuscript_Figures_v1')
    p.add_argument('--manuscript',type=Path,default=ROOT/'paper draft/FTIR_Manuscript.docx')
    args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.manuscript)
