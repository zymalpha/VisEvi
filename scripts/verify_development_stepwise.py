"""Inspect one real-image run: .venv/bin/python scripts/verify_development_stepwise.py --run PATH"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from visevi.core.session import EvidenceSession
from visevi.construction.teachers import ComposeEvidenceCTR
from visevi.eval.runner import build_registry
from visevi.perception.anatomy import LM_GROUPS, load_mask_original_frame
from visevi.perception.geometry import mirror_dice


def positive_ctr_probe():
    session = EvidenceSession(image_ref='synthetic_positive_regression')
    masks = {name: np.zeros((100, 100), bool) for name in ['Heart', 'Right_Lung', 'Left_Lung']}
    masks['Right_Lung'][5:95, 5:40] = True
    masks['Left_Lung'][5:95, 60:95] = True
    masks['Heart'][40:80, 15:85] = True
    session.attach_masks(masks)
    points = np.zeros((120, 2))
    for name, (a, b) in LM_GROUPS.items():
        low, high = {'right_lung': (5, 40), 'left_lung': (60, 94), 'heart': (15, 85)}[name]
        points[a:b, 0] = np.linspace(low, high, b-a)
        points[a:b, 1] = 50
    session.attach_landmarks({'lm_px': points, 'groups': LM_GROUPS})
    claim = ComposeEvidenceCTR().run(session)[0]
    assert claim.value['supported_direction'] == 'present'
    assert '支持方向 present' in session.verdict.answer and '未检出' not in session.verdict.answer
    return {'hand_ctr_mask': '70/90 = 0.7777778', 'claim': claim.value,
            'verdict': session.verdict.to_dict()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sample-key')
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.run/'predictions.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r['status']=='ok' and (not args.sample_key or r['sample_key']==args.sample_key))
    sample = args.run/row['sample_output']
    cache = json.loads((sample/'masks/cache_record.json').read_text())
    image = np.asarray(Image.open(cache['image_path']).convert('L'))
    h,w = image.shape
    masks = {n: load_mask_original_frame(sample/'masks', n, w, h) for n in ['Heart','Right_Lung','Left_Lung']}
    heart, union = masks['Heart'], masks['Right_Lung']|masks['Left_Lung']
    def row_widths(mask):
        return np.array([int(np.where(line)[0][-1]-np.where(line)[0][0]+1) if line.any() else 0 for line in mask])
    heart_widths, thorax_widths = row_widths(heart), row_widths(union)
    heart_y, thorax_y = int(heart_widths.argmax()), int(thorax_widths.argmax())
    heart_w, thorax_w = int(heart_widths[heart_y]), int(thorax_widths[thorax_y])
    assert abs(heart_w/thorax_w-row['measurements']['ctr_max_row_proxy']['CTR'])<1e-12
    ys,xs = np.where(heart)
    x0,x1 = int(xs.min()),int(xs.max())
    ty,tx = np.where(union)
    middle = (int(tx.min())+int(tx.max()))/2
    dr,dl = middle-x0,x1-middle
    assert abs((dr+dl)/thorax_w-row['measurements']['ctr_danzer_proxy']['CTR'])<1e-12
    lung_values = image[union]
    # Recompute real-image reflection with explicit destination columns.
    reflected = np.zeros_like(masks['Right_Lung'])
    for destination_x in range(w):
        source_x = int(round(2 * middle - destination_x))
        if 0 <= source_x < w:
            reflected[:, destination_x] = masks['Right_Lung'][:, source_x]
    mirror_intersection = int((reflected & masks['Left_Lung']).sum())
    mirror_denominator = int(reflected.sum() + masks['Left_Lung'].sum())
    real_dice = 2 * mirror_intersection / mirror_denominator
    assert round(real_dice, 4) == row['measurements']['quality']['rotation']['mirror_dice']
    strip=[]
    for y in range(h):
        a,b = np.where(masks['Right_Lung'][y])[0],np.where(masks['Left_Lung'][y])[0]
        if len(a) and len(b) and b.min()>a.max()+1:
            strip.extend(image[y,a.max()+1:b.min()].tolist())
    exposure = row['measurements']['quality']['exposure_contrast']
    assert round(float(lung_values.mean()), 1) == exposure['lung_mean']
    assert round(float(lung_values.std()), 1) == exposure['lung_std']
    assert round(float(np.mean(strip)), 1) == exposure['mediastinum_strip_mean']
    toy_r,toy_l=np.zeros((10,10),bool),np.zeros((10,10),bool)
    toy_r[2:8,1:4]=True
    toy_l[2:8,6:9]=True
    dice=mirror_dice(toy_r,toy_l,4.5)['mirror_dice']
    assert dice==1.0
    out=args.run/'stepwise'
    if out.exists():
        parser.error(f'{out} exists; previous evidence is never overwritten')
    out.mkdir()
    detail={'sample_key':row['sample_key'],'input_binding':cache['binding'],
        'mask_areas_px':{n:int(m.sum()) for n,m in masks.items()},
        'single_row_ctr':{'heart_width':heart_w,'heart_y':heart_y,'thorax_width':thorax_w,
            'thorax_y':thorax_y,'formula':f'{heart_w}/{thorax_w}','value':heart_w/thorax_w},
        'danzer_proxy':{'midline_x':middle,'heart_x_min':x0,'heart_x_max':x1,
            'd_right':dr,'d_left':dl,'formula':f'({dr}+{dl})/{thorax_w}','value':(dr+dl)/thorax_w},
        'intensity_intermediates':{'lung_count':int(len(lung_values)),
            'lung_sum':int(lung_values.astype('uint64').sum()),'lung_mean':float(lung_values.mean()),
            'lung_std':float(lung_values.std()),'p5_p95':np.percentile(lung_values,[5,95]).tolist(),
            'strip_count':len(strip),'strip_sum':int(sum(strip)),
            'strip_mean':float(np.mean(strip)) if strip else None,
            'clip_zero_count':int((image==0).sum()),'clip_255_count':int((image==255).sum()),
            'total_pixels':int(image.size)},
        'quality_outputs':row['measurements']['quality'],
        'real_image_reflection':{'midline_x':middle,
            'formula':f'2*{mirror_intersection}/{mirror_denominator}',
            'intersection_px':mirror_intersection,
            'reflected_right_area_px':int(reflected.sum()),
            'left_area_px':int(masks['Left_Lung'].sum()),
            'dice':real_dice},
        'synthetic_reflection':{'formula':'x_prime=2*4.5-x; {1,2,3}->{8,7,6}',
            'intersection':18,'area_each':18,'dice':dice},
        'positive_ctr_regression':positive_ctr_probe(),
        'json_tool_menu':{'tool_count':len(build_registry().describe_all()),
            'serialized_bytes':len(json.dumps(build_registry().describe_all()))},
        'ctr_teacher_refusal':row['ctr_teacher_refusal'],
        'limits':['All real-image masks are predictions, not manual GT.',
            'Separate synthetic regressions from real-image clinical performance.']}
    (out/'intermediates.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'row_widths.npz',heart=heart_widths,thorax=thorax_widths)
    lines=['Step 1: input and preprocessing',json.dumps(cache['binding'],ensure_ascii=False,indent=2),
        'Step 2: masks, chosen rows and CTR arithmetic',
        json.dumps({k:detail[k] for k in ['mask_areas_px','single_row_ctr','danzer_proxy']},indent=2),
        'Step 3: intensity intermediates and quality proxies',
        json.dumps({k:detail[k] for k in ['intensity_intermediates','real_image_reflection','quality_outputs']},ensure_ascii=False,indent=2),
        'Step 4: independent synthetic regressions',
        json.dumps({k:detail[k] for k in ['synthetic_reflection','positive_ctr_regression','json_tool_menu']},ensure_ascii=False,indent=2),
        'Step 5: missing independent landmarks -> explicit refusal',
        json.dumps(detail['ctr_teacher_refusal'],ensure_ascii=False,indent=2)]
    text='\n\n'.join(lines)
    (out/'steps.txt').write_text(text,encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    axes[0,0].imshow(image,cmap='gray');axes[0,0].set_title('Real image: predicted geometry')
    for y,mask,color,label in [(heart_y,heart,'red','heart widest row'),(thorax_y,union,'cyan','thorax proxy widest row')]:
        xx=np.where(mask[y])[0]
        axes[0,0].plot([xx.min(),xx.max()],[y,y],color=color,label=label)
    axes[0,0].axvline(middle,color='yellow',linestyle='--');axes[0,0].legend(fontsize=7)
    axes[0,1].imshow(image,cmap='gray')
    colors={'Heart':'red','Right_Lung':'cyan','Left_Lung':'yellow'}
    for name,mask in masks.items():axes[0,1].contour(mask,levels=[0.5],colors=[colors[name]],linewidths=0.7)
    axes[0,1].set_title('Predicted masks in original pixel frame')
    axes[0,2].plot(heart_widths,np.arange(h),label='heart width')
    axes[0,2].plot(thorax_widths,np.arange(h),label='thorax proxy width')
    axes[0,2].invert_yaxis();axes[0,2].legend();axes[0,2].set_title('All per-row widths (inclusive px)')
    axes[1,0].imshow(toy_r.astype(int)+2*toy_l.astype(int),vmin=0,vmax=2,cmap='coolwarm')
    axes[1,0].axvline(4.5,color='black',linestyle='--')
    axes[1,0].set_title(f'Independent reflection check: Dice={dice:.1f}')
    axes[1,1].set_xlim(0,1000);axes[1,1].set_ylim(1000,0)
    axes[1,1].add_patch(plt.Rectangle((200,0),600,1000,fill=False,color='black'))
    axes[1,1].scatter([500],[500],color='green');axes[1,1].axvline(200,linestyle='--')
    axes[1,1].set_title('Letterbox: (500-200,500)=(300,500) original')
    axes[1,2].axis('off')
    q=detail['intensity_intermediates']
    notes=[row['sample_key'],f'Heart row width: {heart_w} @ y={heart_y}',
        f'Thorax proxy width: {thorax_w} @ y={thorax_y}',
        f'Row CTR = {heart_w}/{thorax_w} = {heart_w/thorax_w:.6f}',
        f'Danzer proxy = ({dr:.1f}+{dl:.1f})/{thorax_w}',
        f'  = {(dr+dl)/thorax_w:.6f}',f'Lung mean = {q["lung_sum"]}/{q["lung_count"]}',
        f'  = {q["lung_mean"]:.4f}',f'Strip count = {q["strip_count"]}',
        f'Mirror Dice = 2*{mirror_intersection}/{mirror_denominator}',
        f'  = {real_dice:.6f}',
        'Dual-source CTR teacher: refused', 'Independent landmarks: unavailable',
        'Clinical accuracy: not evaluated']
    axes[1,2].text(0,1,'\n'.join(notes),va='top',family='monospace',fontsize=10)
    fig.suptitle('Development verification: arithmetic and coordinates; no clinical accuracy claim')
    fig.savefig(out/'verification.png',dpi=150)
    plt.close(fig)
    print(text)
    print('Figure:',out/'verification.png')


if __name__=='__main__':
    main()
