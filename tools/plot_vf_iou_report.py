"""Export faithful AP charts and geometric 3D IoU examples from saved VF outputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/vf_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from vf_iou3d import _geometry, iou3d_matrix

THRESHOLDS = ['0.25', '0.5', '0.7']
COLORS = ['#e58c16', '#2676c5', '#8753b5']
LABELS = ['BEVFusion', 'BEVFusion + DetZero (original)', 'DetZero + motorcycle bottom anchor*']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/vf-car-truck-sweeps2-01')
    args = parser.parse_args()
    run = Path(args.run_dir)
    out = run/'iou_report_figures'
    out.mkdir(parents=True, exist_ok=True)
    inputs = [run/'comparison_iou3d.json', run/'z_fix/counterfactual_metrics.json',
              run/'baseline/results_nusc_center_corrected.json', run/'dataset_local/v1.0-mini/sample_annotation.json',
              run/'visual_overlay/manifest.json']
    before_after = json.loads(inputs[0].read_text())
    anchored = json.loads(inputs[1].read_text())['motorcycle_bottom']
    metrics = [before_after['baseline'], before_after['refined'], anchored]
    assert all(m['car_includes_truck'] for m in metrics)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42})
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.3), sharey=True)
    for ax, cls, title in zip(axes, ['car','motorcycle'], ['Car (including truck)', 'Motorcycle']):
        x = np.arange(3)
        n = 2 if cls == 'car' else 3
        width = .27 if n == 2 else .22
        for j in range(n):
            values = [100*metrics[j]['iou3d']['per_class'][cls]['by_threshold'][t]['AP'] for t in THRESHOLDS]
            bars = ax.bar(x+(j-(n-1)/2)*width, values, width, color=COLORS[j], zorder=3)
            for bar, value, t in zip(bars, values, THRESHOLDS):
                label = '~0' if 0 < value < .001 else f'{value:.2f}'
                ax.text(bar.get_x()+bar.get_width()/2, value+1.5, label,
                        ha='center', va='bottom', fontsize=10, fontweight='medium')
                rows.append({'class':cls,'method':LABELS[j],'IoU_threshold':t,'AP_percent':value})
        ax.set_title(title, fontsize=15, fontweight='bold', pad=16)
        ax.set_xticks(x, ['0.25','0.50','0.70'])
        ax.set_xlabel('3D IoU matching threshold', labelpad=10)
        ax.set_ylim(0, 102)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#e2e6eb')
        ax.spines['left'].set_color('#d0d5db'); ax.spines['bottom'].set_color('#d0d5db')
    axes[0].set_ylabel('Average Precision (%)')
    fig.suptitle('Detection AP under 3D IoU matching', fontsize=20, fontweight='bold', y=.98)
    fig.legend([Patch(color=c) for c in COLORS], LABELS, loc='lower center',
               bbox_to_anchor=(.5,.112), ncol=3, frameon=False, fontsize=10)
    fig.text(.055,.068,'VF adapted | 156 frames | Custom full-recall AP | Original DetZero input threshold = 0.1',fontsize=9,color='#47515c')
    fig.text(.055,.037,'*Bottom-anchor results use saved GRM/PRM predictions; models and tracker were not rerun for these AP values.',fontsize=9,color='#47515c')
    fig.text(.055,.012,'Car AP is unchanged by motorcycle-only anchoring; duplicate bars are omitted. Matching uses class, confidence order and one-to-one GT assignment.',fontsize=8.4,color='#47515c')
    fig.subplots_adjust(top=.81,bottom=.27,left=.075,right=.985,wspace=.12)
    for ext in ['png','pdf','svg']:
        fig.savefig(out/('ap_vs_iou_car_motorcycle.'+ext), dpi=240, facecolor='white')
    plt.close(fig)
    with (out/'ap_values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Geometric examples use unchanged source boxes, not median synthetic shapes.
    baseline = json.loads(inputs[2].read_text())['results']
    annotations = {r['token']:r for r in json.loads(inputs[3].read_text())}
    manifest = json.loads(inputs[4].read_text())
    selections = [next(f for f in manifest['frames'] if f['index']==30 and f['focus']['class']=='car'),
                  next(f for f in manifest['frames'] if f['index']==80 and f['focus']['class']=='motorcycle')]
    fig = plt.figure(figsize=(13,7.4))
    examples=[]
    for index, frame in enumerate(selections):
        focus=frame['focus']; gt=annotations[focus['gt_token']]; pred=baseline[frame['sample_token']][int(focus['source_index'])]
        def box(record):return SimpleNamespace(**{k:record[k] for k in ['translation','size','rotation']})
        iou=float(iou3d_matrix([box(pred)],[box(gt)])[0,0])
        gt_poly,gb,gtop,gv=_geometry(box(gt)); pp,pb,ptop,pv=_geometry(box(pred))
        polygon=gt_poly.intersection(pp); low=max(gb,pb); high=min(gtop,ptop)
        volume=polygon.area*max(0,high-low);union=gv+pv-volume
        assert abs(iou-volume/union)<1e-9
        rotation=Quaternion(gt['rotation']).rotation_matrix;center=np.array(gt['translation'])
        local=lambda points:(points-center)@rotation
        ax=fig.add_subplot(1,2,index+1,projection='3d')
        allpoints=[]
        for record,color in [(gt,'#36a65b'),(pred,COLORS[0])]:
            pts=local(Box(record['translation'],record['size'],Quaternion(record['rotation'])).corners().T)
            allpoints.append(pts)
            faces=[pts[[0,1,2,3]],pts[[4,5,6,7]],pts[[0,1,5,4]],pts[[2,3,7,6]],pts[[0,3,7,4]],pts[[1,2,6,5]]]
            ax.add_collection3d(Poly3DCollection(faces,facecolor=color,edgecolor=color,alpha=.12,linewidth=.7))
            edges=[(i,i+4) for i in range(4)]+[(i,(i+1)%4) for i in range(4)]+[(i+4,(i+1)%4+4) for i in range(4)]
            for a,b in edges:ax.plot(pts[[a,b],0],pts[[a,b],1],pts[[a,b],2],color=color,linewidth=1.6)
        if not polygon.is_empty and high>low:
            xy=np.array(polygon.exterior.coords)[:-1]
            bottom=local(np.column_stack([xy,np.full(len(xy),low)]))
            top=local(np.column_stack([xy,np.full(len(xy),high)]))
            faces=[bottom,top]+[np.array([bottom[i],bottom[(i+1)%len(xy)],top[(i+1)%len(xy)],top[i]]) for i in range(len(xy))]
            ax.add_collection3d(Poly3DCollection(faces,facecolor='#64748b',edgecolor='#64748b',alpha=.22,linewidth=.5))
        points=np.vstack(allpoints);lower=points.min(axis=0)-.15;upper=points.max(axis=0)+.15
        ax.set_xlim(lower[0],upper[0]);ax.set_ylim(lower[1],upper[1]);ax.set_zlim(lower[2],upper[2]);ax.set_box_aspect(upper-lower)
        ax.view_init(elev=22,azim=-58)
        ax.set_xlabel('GT local X (m)',labelpad=8);ax.set_ylabel('GT local Y (m)',labelpad=8);ax.set_zlabel('GT local Z (m)',labelpad=8)
        ax.tick_params(labelsize=8)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_major_locator(MaxNLocator(nbins=5))
        name='Car' if focus['class']=='car' else 'Motorcycle'
        ax.set_title(f'{name}: IoU 3D = {iou:.3f}\n'+('Passes' if iou>=.5 else 'Fails')+' the 0.50 geometric threshold',fontsize=14,fontweight='bold',pad=14)
        fig.text(.075+index*.49,.20,f'Frame {frame["index"]} | source index {focus["source_index"]}\nIntersection: {volume:.3f} m³ | Union: {union:.3f} m³\nPrediction / GT volume: {pv/gv:.2f}x',fontsize=10)
        examples.append({'class':focus['class'],'frame':frame['index'],'sample_token':frame['sample_token'],
                         'source_index':int(focus['source_index']),'gt_token':focus['gt_token'],'GT':gt,'prediction':pred,
                         'iou3d':iou,'intersection_m3':volume,'union_m3':union,'prediction_to_GT_volume':pv/gv})
    fig.suptitle('Why a detected object can fail 3D IoU matching',fontsize=19,fontweight='bold',y=.985)
    fig.legend([Patch(facecolor='#36a65b'),Patch(facecolor=COLORS[0]),Patch(facecolor='#64748b')],
               ['Ground truth','BEVFusion detection','3D intersection'],loc='lower center',bbox_to_anchor=(.5,.105),ncol=3,frameon=False)
    fig.text(.075,.074,'IoU 3D = intersection volume / union volume. Actual saved box coordinates are used; no manual alignment.',fontsize=9,color='#47515c')
    fig.text(.075,.046,'Examples are selected diagnostic pairs, not population averages. Passing IoU alone does not establish a TP without class/score/assignment checks.',fontsize=8.3,color='#47515c')
    fig.subplots_adjust(top=.82,bottom=.29,left=.035,right=.97,wspace=.02)
    for ext in ['png','pdf','svg']:fig.savefig(out/('iou3d_geometry_car_motorcycle.'+ext),dpi=240,facecolor='white')
    plt.close(fig)
    provenance={'run':str(run),'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'examples':examples,
                'AP_policy':'Custom VF AP, car includes truck, original DetZero min_score .1; bottom-anchor AP from frozen saved outputs'}
    (out/'sources.json').write_text(json.dumps(provenance,indent=2))
    (out/'README.md').write_text('''# Report figures

English labels, generated directly from saved run data. PNG (240 dpi), vector PDF and SVG exports are included.

- `ap_vs_iou_car_motorcycle`: AP (%) at matching thresholds 0.25 / 0.50 / 0.70. AP is not mean IoU. Car includes truck. Motorcycle-only bottom anchoring uses frozen saved model outputs, not a complete rerun. Car duplicate anchor bars are omitted.
- `iou3d_geometry_car_motorcycle`: true 3D GT / BEVFusion boxes and their volumetric intersection for two selected saved pairs. Axes are in each GT object's local coordinate frame, in meters. Box positions, dimensions and yaw are unchanged. Geometric threshold pass/fail is not a complete TP/FP determination.
- `ap_values.csv` and `sources.json`: numeric values and exact source boxes/hashes.

Reproduce: `python3 tools/plot_vf_iou_report.py` from repository root.
''')
    print('Saved',out)
    print('Examples:',[(e['class'],e['iou3d']) for e in examples])


if __name__=='__main__':main()
