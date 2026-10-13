"""Build the English LaTeX report and charts from an actual corrected DetZero run."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/vf_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

THRESHOLDS = ['0.25', '0.5', '0.7']
CLASSES = ['car', 'motorcycle', 'pedestrian']
NAMES = {'car': 'Car (including truck)', 'motorcycle': 'Motorcycle', 'pedestrian': 'Pedestrian'}
COLORS = ['#e18b18', '#269065']
METHODS = ['BEVFusion', 'BEVFusion + DetZero']


def read(p): return json.loads(p.read_text())
def ap(m,c,t): return m['iou3d']['per_class'][c]['by_threshold'][t]['AP']*100

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/vf-car-truck-sweeps2-01')
    parser.add_argument('--corrected-dir', default='grm_prm_zfix_OTAgmB')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];run=(root/args.run_dir).resolve();fixed=run/args.corrected_dir
    out=fixed/'report';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    new=read(fixed/'comparison_iou3d.json');report=read(fixed/'run_report.json')
    metrics=[new['baseline'], new['refined']]
    assert all(m['samples']==156 and m['car_includes_truck'] for m in metrics)
    for m in metrics:
        for k in ['classes','range_m','sample_tokens_sha256','include_zero_points','car_includes_truck']:
            assert m[k]==metrics[0][k],k
        for c in CLASSES: assert m['per_class'][c]['gt_boxes']==metrics[0]['per_class'][c]['gt_boxes']
    assert report['input_sha256']==metrics[0]['prediction_sha256']
    assert report['arguments']['preserve_bottom_classes']=='motorcycle' and report['arguments']['min_score']==.1
    pred=read(fixed/'results_nusc_detzero_refined.json');baseline=read(run/'baseline/results_nusc_center_corrected.json')
    assert hashlib.sha256((fixed/'results_nusc_detzero_refined.json').read_bytes()).hexdigest()==metrics[1]['prediction_sha256']
    anchored=[];total=0
    for token, records in baseline['results'].items():
        assert len(records)==len(pred['results'][token])
        for a,b in zip(records,pred['results'][token]):
            assert a['sample_token']==b['sample_token']==token
            assert a['detection_name']==b['detection_name'] and a['detection_score']==b['detection_score']
            total+=1
            if a['detection_name']=='motorcycle' and abs(a['size'][2]-b['size'][2])>1e-6:
                anchored.append(b['translation'][2]-b['size'][2]/2-(a['translation'][2]-a['size'][2]/2))
    assert max(abs(v) for v in anchored)<1e-6
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(figs/(name+'.'+ext),dpi=260,facecolor='white')
        plt.close(fig)
    def label(value):return '~0' if 0<value<.001 else f'{value:.2f}'
    def style(ax):
        ax.set_axisbelow(True);ax.yaxis.grid(color='#e1e6eb');ax.spines['left'].set_color('#c9d0d6');ax.spines['bottom'].set_color('#c9d0d6')
    fig, axes=plt.subplots(1,3,figsize=(15,6.3))
    for ax,c,limit in zip(axes,CLASSES,[110,96,8.7]):
        for j,m in enumerate(metrics):
            values=[ap(m,c,t) for t in THRESHOLDS]
            bars=ax.bar(np.arange(3)+(j-.5)*.30,values,.30,color=COLORS[j],label=METHODS[j],zorder=3)
            for k,(b,v) in enumerate(zip(bars,values)):
                close = any(abs(v-ap(metrics[earlier],c,THRESHOLDS[k])) < limit*.035 for earlier in range(j))
                ax.annotate(label(v),(b.get_x()+b.get_width()/2,v),xytext=(0,6+10*j if close else 6),
                            textcoords='offset points',ha='center',va='bottom',fontsize=8.4)
        ax.set(title=NAMES[c],xticks=np.arange(3),xticklabels=['0.25','0.50','0.70'],ylim=(0,limit),xlabel='3D IoU threshold')
        ax.set_ylabel('AP (%)');style(ax)
    fig.suptitle('Detection AP: BEVFusion vs. BEVFusion + DetZero',fontsize=18,fontweight='bold',y=.97)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.12),ncol=2,frameon=False)
    fig.text(.055,.065,'Actual GPU rerun: grm_prm_zfix_OTAgmB | Same 156 frames and detector baseline | DetZero input threshold = 0.1',fontsize=9,color='#46525d')
    fig.text(.055,.031,'Panel scales differ. Car includes truck. DetZero uses measurement-based reverse tracking and motorcycle bottom anchoring; no CRM.',fontsize=9,color='#46525d')
    fig.subplots_adjust(top=.81,bottom=.28,left=.06,right=.985,wspace=.22);save(fig,'per_class_ap')
    fig,ax=plt.subplots(figsize=(10.8,5.7))
    for j,m in enumerate(metrics):
        values=[100*m['iou3d']['mAP_by_threshold'][t] for t in THRESHOLDS]
        bars=ax.bar(np.arange(3)+(j-.5)*.30,values,.30,color=COLORS[j],label=METHODS[j],zorder=3)
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.2f}',ha='center',fontsize=10)
    ax.set(xticks=np.arange(3),xticklabels=['0.25','0.50','0.70'],ylim=(0,68),xlabel='3D IoU threshold',ylabel='mAP (%)')
    style(ax);fig.suptitle('Overall mAP improves most at IoU 0.50',fontsize=18,fontweight='bold',y=.97)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,frameon=False,fontsize=9.5)
    fig.text(.075,.045,'mAP = equal-weight mean of car, motorcycle and pedestrian AP. DetZero values come from the actual GPU run.',fontsize=8.5,color='#46525d')
    fig.subplots_adjust(top=.85,bottom=.26,left=.09,right=.985);save(fig,'overall_map')
    fig,axes=plt.subplots(1,3,figsize=(15,6.1))
    for ax,t in zip(axes,THRESHOLDS):
        values=[ap(metrics[1],c,t)-ap(metrics[0],c,t) for c in CLASSES]
        bars=ax.bar(np.arange(3),values,.5,color=COLORS[1],label=METHODS[1])
        for b,v in zip(bars,values):
            ax.annotate(f'{v:+.2f}',(b.get_x()+b.get_width()/2,v),xytext=(0,5 if v>=0 else -12),
                        textcoords='offset points',ha='center',fontsize=9)
        ax.set(xticks=np.arange(3),xticklabels=['Car','Motorcycle','Pedestrian'],ylim=(-3,46),title='IoU '+('0.50' if t=='0.5' else t))
        ax.set_ylabel('AP gain vs. BEVFusion (percentage points)');ax.axhline(0,color='#444d57',linewidth=1);style(ax)
    fig.suptitle('DetZero effect: class-specific AP gains and remaining regression',fontsize=17,fontweight='bold',y=.97)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.10),ncol=2,frameon=False)
    fig.text(.06,.047,'Same vertical scale across panels. At IoU 0.50, the DetZero gains are +12.53 pp for car, +41.43 pp for motorcycle, and -0.93 pp for pedestrian.',fontsize=9,color='#46525d')
    fig.subplots_adjust(top=.81,bottom=.26,left=.06,right=.985,wspace=.23);save(fig,'ap_gains')
    fig,ax=plt.subplots(figsize=(11.5,5.8))
    values=[(NAMES[c],ap(metrics[0],c,'0.5'),ap(metrics[1],c,'0.5')) for c in CLASSES]
    values.append(('Overall mAP',100*metrics[0]['iou3d']['mAP_by_threshold']['0.5'],100*metrics[1]['iou3d']['mAP_by_threshold']['0.5']))
    for i,(name,before,after) in enumerate(values):
        y=3-i;color=COLORS[1] if after>=before else '#ba4141'
        ax.plot([before,after],[y,y],color=color,linewidth=5,alpha=.4,solid_capstyle='round')
        ax.scatter(before,y,s=85,color=COLORS[0],zorder=4)
        ax.scatter(after,y,s=85,color=color,zorder=5)
        ax.annotate(f'{before:.2f}%',(before,y),xytext=(0,-20),textcoords='offset points',ha='center',fontsize=10)
        ax.annotate(f'{after:.2f}%',(after,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10,fontweight='bold')
        ax.text(71,y,f'{after-before:+.2f} pp',color=color,fontsize=12,va='center',fontweight='bold')
    ax.set(yticks=[3,2,1,0],yticklabels=[v[0] for v in values],ylim=(-.55,3.65),xlim=(-3,83),xlabel='AP / mAP (%)')
    ax.xaxis.grid(color='#e1e6eb');ax.set_axisbelow(True);ax.spines['left'].set_visible(False)
    ax.set_xticks(np.arange(0,71,10));ax.tick_params(axis='y',length=0,labelsize=11)
    ax.scatter([],[],color=COLORS[0],label='BEVFusion baseline');ax.scatter([],[],color=COLORS[1],label='BEVFusion + DetZero')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.085),ncol=2,frameon=False)
    fig.suptitle('At IoU 0.50: the direct effect of DetZero',fontsize=18,fontweight='bold',y=.97)
    fig.text(.065,.035,'Same detector baseline, class mapping and evaluation protocol | No CRM | Red indicates pedestrian regression | pp = percentage points',fontsize=8.6,color='#46525d')
    fig.subplots_adjust(top=.84,bottom=.26,left=.19,right=.97);save(fig,'ap50_before_after')
    rows=[]
    for t in THRESHOLDS:
        for c in CLASSES:
            rows.append({'IoU':t,'class':c,'baseline_AP_percent':ap(metrics[0],c,t),
                         'detzero_AP_percent':ap(metrics[1],c,t),'gain_vs_baseline_pp':ap(metrics[1],c,t)-ap(metrics[0],c,t)})
    with (out/'ap_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    provenance={'actual_run':str(fixed.relative_to(root)),'checkpoint_hashes':{k:v['sha256'] for k,v in report['checkpoints'].items()},'evaluation_policy_matches':True,
                'score_and_class_preserved_boxes':total,'motorcycle_height_changed_boxes':len(anchored),
                'max_motorcycle_bottom_change_m':max(abs(v) for v in anchored),
                'input_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [fixed/'comparison_iou3d.json',fixed/'run_report.json',fixed/'results_nusc_detzero_refined.json']}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    # Construct English report from validated numeric fields, never pasted rounded values.
    def fmt(v):
        return ('$'+f'{v/10**np.floor(np.log10(v)):.4f}'+r'\times 10^{'+str(int(np.floor(np.log10(v))))+'}$') if 0<v<.0001 else f'{v:.4f}'
    def row(values):return ' & '.join(values)+r' \\'+'\n'
    maprows=''
    for t in THRESHOLDS:
        vals=[m['iou3d']['mAP_by_threshold'][t]*100 for m in metrics]
        maprows+=row([f'{float(t):.2f}']+[fmt(v) for v in vals]+[f'{vals[1]-vals[0]:+.4f}'])
    classrows=''
    for v in rows:
        classrows+=row([f"{float(v['IoU']):.2f}",NAMES[v['class']]]+[fmt(v[k]) for k in ['baseline_AP_percent','detzero_AP_percent']]+[f"{v['gain_vs_baseline_pp']:+.4f}"])
    countrows=''
    for t in THRESHOLDS:
        for c in CLASSES:
            vals=[m['iou3d']['per_class'][c]['by_threshold'][t] for m in [metrics[0],metrics[1]]]
            countrows+=row([f'{float(t):.2f}',NAMES[c]]+[str(m[k]) for k in ['TP','FP','FN'] for m in vals])
    errors=''
    for c in CLASSES:
        errors+=row([NAMES[c]]+[f"{m['per_class'][c][k]:.4f}" for k in ['trans_err','scale_err','orient_err'] for m in [metrics[0],metrics[1]]])
    def figure(name,caption,label):
        return '\\begin{figure}[H]\n\\centering\n\\reportfigure{'+name+'}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{figure}\n'
    tex=LATEX_TEMPLATE.replace('@@MAP_ROWS@@',maprows).replace('@@CLASS_ROWS@@',classrows).replace('@@COUNT_ROWS@@',countrows).replace('@@ERROR_ROWS@@',errors)
    tex=tex.replace('@@MAP_FIGURE@@',figure('overall_map','Overall custom 3D IoU mAP for BEVFusion and BEVFusion + DetZero, using the actual supplied run.','fig:map'))
    tex=tex.replace('@@CLASS_FIGURE@@',figure('per_class_ap','Class AP at each IoU threshold. Panel scales differ; compare methods within a class. Car includes truck.','fig:classes'))
    tex=tex.replace('@@GAIN_FIGURE@@',figure('ap_gains','AP gains relative to BEVFusion, in percentage points. The common vertical scale makes the large motorcycle gain and pedestrian regression explicit.','fig:gains'))
    tex=tex.replace('@@AP50_FIGURE@@',figure('ap50_before_after','Direct BEVFusion-to-DetZero comparison at IoU 0.50. Labels are absolute AP/mAP percentages; right-hand differences are percentage points. Red highlights pedestrian regression.','fig:ap50'))
    for key,value in {'@@BASE_HASH@@':metrics[0]['prediction_sha256'],'@@NEW_HASH@@':metrics[1]['prediction_sha256'],
                      '@@GIT_COMMIT@@':report['git_commit'],'@@PROCESSED@@':str(report['status_counts']['processed']),
                      '@@TRACKS@@':str(report['tracks']),'@@REPLACED@@':str(report['matched_boxes_replaced'])}.items():tex=tex.replace(key,value)
    assert '@@' not in tex
    docs=root/'docs'; docassets=docs/'figures/vf_zfix';docassets.mkdir(parents=True,exist_ok=True)
    for name in ['overall_map','per_class_ap','ap_gains','ap50_before_after']:shutil.copy2(figs/(name+'.pdf'),docassets/(name+'.pdf'))
    (docs/'BEVFUSION_DETZERO_IOU3D_REPORT.tex').write_text(tex)
    (out/'main.tex').write_text(tex)
    (out/'README.md').write_text('''# BEVFusion vs. BEVFusion + DetZero

DetZero run: `grm_prm_zfix_OTAgmB`. English LaTeX uses pdfLaTeX and ships with vector PDF figures.

Upload `OVERLEAF_REPORT.zip` as a new Overleaf project; choose `main.tex` and pdfLaTeX. The figure paths in the ZIP are complete. Uploading only the .tex will show figure placeholders; use the full ZIP to display charts.

The charts compare only BEVFusion and BEVFusion + DetZero for all three classes. Car includes truck. Score/classes/order and checkpoint identities were checked; DetZero refers to the actual supplied run.

Reproduce: `python3 tools/build_vf_zfix_report.py` from repository root. No TeX compiler was available locally; the .tex has not been compiled here.
''')
    with zipfile.ZipFile(out/'OVERLEAF_REPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(out/'main.tex','main.tex')
        for name in ['overall_map','per_class_ap','ap_gains','ap50_before_after']:z.write(figs/(name+'.pdf'),'figures/vf_zfix/'+name+'.pdf')
        for name in ['README.md','ap_comparison.csv','provenance.json']:z.write(out/name,name)
    with zipfile.ZipFile(out/'REPORT_AND_CHARTS.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['main.tex','OVERLEAF_REPORT.zip','README.md','ap_comparison.csv','provenance.json']:
            z.write(out/name,name)
        if (out/'metrics_recomputed_verified.json').exists():
            z.write(out/'metrics_recomputed_verified.json','metrics_recomputed_verified.json')
        for path in figs.iterdir():
            if path.is_file():z.write(path,'figures/'+path.name)
        z.write(Path(__file__),Path(__file__).name)
    print('Actual-run report saved:',out)
    print('Updated:',docs/'BEVFUSION_DETZERO_IOU3D_REPORT.tex')


LATEX_TEMPLATE=r'''% Compile with pdfLaTeX. Upload the full OVERLEAF_REPORT.zip for all figures.
\documentclass[a4paper,11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,booktabs,array,graphicx,float}
\usepackage[hidelinks]{hyperref}
\usepackage{seqsplit}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\renewcommand{\arraystretch}{1.15}
\newcommand{\reportfigure}[1]{\IfFileExists{figures/vf_zfix/#1.pdf}{\includegraphics[width=\linewidth]{figures/vf_zfix/#1.pdf}}{\fbox{\parbox{0.94\linewidth}{Figure file missing: upload the complete OVERLEAF\_REPORT.zip, including figures/vf\_zfix/.}}}}
\title{BEVFusion and DetZero Detection Evaluation\\\large 3D IoU AP at Thresholds 0.25, 0.50 and 0.70}
\author{VF Adapted Dataset Experiment}
\date{October 13, 2026}
\begin{document}
\maketitle
\begin{abstract}
This report compares BEVFusion with BEVFusion + DetZero on the same 156 VF adapted frames. At 3D IoU 0.50, mAP increases from 16.9240\% to 34.6021\%, a gain of 17.6781 percentage points. Car AP increases from 49.2058\% to 61.7406\%, and motorcycle AP rises from 0.5390\% to 41.9702\%. Pedestrian AP decreases from 1.0271\% to 0.0957\% and remains unresolved. DetZero uses GRM/PRM without CRM, measurement-based single-stage reverse tracking, and motorcycle-bottom anchoring when combining dimensions and pose. Scores are unchanged. Results follow a custom full-recall 3D IoU AP protocol, not official Waymo or nuScenes metrics.
\end{abstract}

\section{Experiment and Verified Comparability}
In this report, \textbf{DetZero} refers to the actual run \texttt{grm\_prm\_zfix\_OTAgmB}, using the final pipeline configuration described below. Its prediction JSON, evaluation metrics and run report were supplied from the company machine.

The two methods use the same detector baseline, sample tokens, GT counts, class mapping, range filters and zero-point exclusion policy. Prediction identifiers were checked against the saved files. All 21,400 exported detection records preserve their source class, confidence and order.

\begin{table}[H]\centering\small
\caption{Evaluated counts after shared class and range filtering.}
\begin{tabular}{lrrr}\toprule
Class & GT & BEVFusion & DetZero \\
\midrule
Car (including truck) & 4873 & 8808 & 8806 \\
Motorcycle & 2492 & 2650 & 2650 \\
Pedestrian & 520 & 7036 & 7036 \\
\bottomrule\end{tabular}\end{table}
Evaluation ranges are 50 m for car and 40 m for motorcycle and pedestrian. The shared exclusions are 903 zero-point and 6,687 out-of-range GT boxes. Car and truck remain separate during tracking/refining and are merged into the Car evaluation group. Partner GT point counts are used without recomputation.

The DetZero run uses input confidence threshold 0.1, two-sweep detector baseline, global coordinates, CPU tracking, CUDA GRM/PRM, \texttt{size\_policy=grm}, and \texttt{z\_policy=center} with \texttt{preserve\_bottom\_classes=motorcycle}. It has @@TRACKS@@ tracks, @@PROCESSED@@ processed tracks, one insufficient-point track, and @@REPLACED@@ replaced source observations. It does not fill missed detections and does not run CRM. The input threshold is not an output evaluation cutoff.

\section{IoU and AP Protocol}
Boxes use global geometric centers, width--length--height dimensions and yaw rotation around global Z. Rotated BEV intersection area is multiplied by vertical overlap:
\begin{equation}
\operatorname{IoU}_{3D}(B_p,B_g)=\frac{V(B_p\cap B_g)}{V(B_p)+V(B_g)-V(B_p\cap B_g)}.
\end{equation}
Predictions are sorted by descending confidence and greedily matched to the best unmatched GT of the same class and frame. A match requires $\operatorname{IoU}_{3D}\geq\tau$; each GT can match once. AP integrates the interpolated precision envelope over the full recall interval $[0,1]$. The IoU evaluator uses all candidates remaining after class/range filters, with no output confidence cutoff. Thus AP is neither the mean IoU nor simply the fraction of TP boxes. Differences below are percentage points (pp).

\section{Overall mAP Improvement}
\begin{table}[H]\centering\small
\caption{Custom 3D IoU mAP (\%) and DetZero gains (pp).}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{rrrr}\toprule
IoU & BEVFusion & BEVFusion + DetZero & $\Delta$ (pp) \\
\midrule
@@MAP_ROWS@@\bottomrule\end{tabular}\end{table}
At IoU 0.50, DetZero improves mAP by 17.6781 pp relative to BEVFusion. mAP is the equal-weight mean of the three class AP values. The strict IoU 0.70 result remains low despite improvement, so the correction does not establish high-overlap accuracy for all boxes.
@@AP50_FIGURE@@
@@MAP_FIGURE@@

\section{Per-Class AP and DetZero Effects}
\begin{table}[H]\centering\scriptsize
\caption{Per-class AP (\%) at each 3D IoU threshold. Gains are pp.}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{rlrrr}\toprule
IoU & Class & BEVFusion & BEVFusion + DetZero & $\Delta$ (pp) \\
\midrule
@@CLASS_ROWS@@\bottomrule\end{tabular}\end{table}
Small nonzero values are retained in the table; plot labels close to zero are rounded. Baseline motorcycle AP at IoU 0.70 is about $1.6460\times10^{-5}\%$; pedestrian AP is about 0.0003282\% for all methods.
@@CLASS_FIGURE@@
@@GAIN_FIGURE@@

\subsection{Car}
At IoU 0.50, car AP improves from 49.2058\% to 61.7406\%, a gain of 12.5347 pp. At IoU 0.25, AP increases from 89.3163\% to 89.5836\%; at IoU 0.70 it increases from 2.2945\% to 4.9117\%. Strict-overlap performance remains limited.

\subsection{Motorcycle}
At IoU 0.50, motorcycle AP increases from 0.5390\% to 41.9702\%, a gain of 41.4311 pp. At IoU 0.25 and 0.70, DetZero AP reaches 75.6851\% and 2.7400\%, respectively.

A diagnostic analysis of 1,563 fixed baseline/GT proximity pairs (baseline score at least 0.1 and same evaluation class, XY distance below 0.5 m) found baseline median width, length and height ratios of approximately 1.32, 1.41 and 1.23 relative to GT. Median volume ratio was 2.29, and 75.2\% of these pairs exceeded twice the GT volume. Since intersection is at most GT volume and union is at least prediction volume, $\operatorname{IoU}_{3D}\leq V_g/V_p$ when $V_p\geq V_g$. Oversized boxes can therefore fail IoU 0.50 even at accurate centers. These proximity pairs are diagnostic and are not independently verified identities in crowded scenes.

\subsection{Pedestrian}
Pedestrian AP at IoU 0.50 remains 0.0957\%, below baseline 1.0271\%. The motorcycle-only bottom policy does not address this regression. Earlier stage diagnostics identified width/length shrinkage at GRM as a major issue. Class-specific size-policy evaluation and annotation-definition checks remain necessary before applying broader changes.

\section{Why the Z Correction Helps}
For upright boxes,
\begin{equation}
z_{\mathrm{bottom}}=z_{\mathrm{center}}-h/2,\qquad
\Delta z_{\mathrm{bottom}}=\Delta z_{\mathrm{center}}-\Delta h/2.
\end{equation}
Replacing height while retaining center Z can shift the box bottom. Motorcycle-bottom anchoring prevents an additional vertical displacement caused solely by the change in GRM height.

For motorcycle, the DetZero combination is
\begin{equation}
z_{\mathrm{new}}=\left(z_{\mathrm{detector}}-h_{\mathrm{detector}}/2\right)+h_{\mathrm{GRM}}/2.
\end{equation}
XY and yaw remain from PRM; the bottom policy replaces PRM Z for the selected class. Across 1,588 motorcycle predictions with changed height in the actual DetZero output, the maximum absolute bottom change relative to the detector is about $1.2\times10^{-7}$ m. This confirms bottom anchoring numerically; it is not an independently estimated road plane and preserves any baseline bottom error.

The reverse-tracking repair also updates a single-stage match from its measurement instead of retaining a predicted box while claiming its detection source index. Both changes are present in the DetZero run. Their individual causal contributions cannot be isolated from a comparison of the combined run alone.

\section{TP, FP and FN}
\begin{table}[H]\centering\small
\caption{All-candidate counts: BEVFusion (B) versus DetZero (D).}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{rlrrrrrr}\toprule
& & \multicolumn{2}{c}{TP} & \multicolumn{2}{c}{FP} & \multicolumn{2}{c}{FN} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
IoU & Class & B & D & B & D & B & D \\
\midrule
@@COUNT_ROWS@@\bottomrule\end{tabular}\end{table}
At IoU 0.50, motorcycle TP count rises from 171 to 1,222. Car gains 433 TP, motorcycle gains 1,051 TP, and pedestrian loses 29 TP relative to baseline. All exported detection scores/classes/counts are preserved. Fewer FP primarily reflect changed geometric matches, not removal of detections. Two car candidates cross the evaluation range boundary after refinement.

\section{Center-Distance Metrics}
\begin{table}[H]\centering\small
\caption{Center-distance evaluator TP errors: BEVFusion versus DetZero; lower is better.}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrr}\toprule
& \multicolumn{2}{c}{ATE (m)} & \multicolumn{2}{c}{ASE} & \multicolumn{2}{c}{AOE (rad)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Class & Baseline & DetZero & Baseline & DetZero & Baseline & DetZero \\
\midrule
@@ERROR_ROWS@@\bottomrule\end{tabular}\end{table}
Center-distance mAP changes from 53.9100\% to 54.0852\%, while IoU 0.50 mAP gains 17.6781 pp. The protocols use different matching and AP integration, so absolute scores are not directly comparable. Near-constant motorcycle translation error with improved scale/orientation error is consistent with stronger geometric overlap. Error TP sets can differ between methods; this table is not a fixed-pair analysis.

\section{Limitations and Conclusions}
The DetZero run supports a substantial improvement in car and motorcycle geometry without CRM. Pedestrian regression and low AP at strict IoU 0.70 remain. Results cover one scene with 156 frames; they do not establish generalization to new scenes. Class definitions, especially rider/motorcycle annotation extent, and the quality of sensor calibration and GT should be checked on independent data. The custom VF evaluator does not report official Waymo/nuScenes mAP/mAPH, NDS, or tracking metrics.

The report verifies saved metrics against prediction identifiers and common evaluation policy. Checkpoint hashes were compared from run reports; a separate new GRM/PRM inference was not performed locally. DetZero results in this report are from the supplied actual company-machine run.

\section{Data Provenance}
Source comparison: \path{outputs/vf-car-truck-sweeps2-01/grm_prm_zfix_OTAgmB/comparison_iou3d.json}. DetZero prediction SHA-256 was checked against its file. The accompanying CSV and provenance JSON contain numeric values and verification details.

\begin{itemize}
\item Baseline prediction SHA-256: {\footnotesize\ttfamily\seqsplit{@@BASE_HASH@@}}.
\item DetZero prediction SHA-256: {\footnotesize\ttfamily\seqsplit{@@NEW_HASH@@}}.
\item DetZero run git commit: {\footnotesize\ttfamily\seqsplit{@@GIT_COMMIT@@}}.
\end{itemize}
\end{document}
'''

if __name__=='__main__':main()
