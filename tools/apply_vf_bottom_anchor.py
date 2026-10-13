"""Apply class-specific detector-bottom anchoring to saved refinement outputs.

No GT access or model inference. This isolates the Z/height combination policy;
it does NOT replay the repaired tracker or recompute GRM/PRM features.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def anchor_saved(baseline, refined, classes):
    if set(baseline['results']) != set(refined['results']):
        raise ValueError('Baseline/refined sample tokens differ')
    output = copy.deepcopy(refined)
    changed = 0
    for token, records in output['results'].items():
        original = baseline['results'][token]
        if len(records) != len(original):
            raise ValueError('Requires source-index-preserving detection outputs; no gap-filled JSON')
        for before, after in zip(original, records):
            if (before['sample_token'] != token or after['sample_token'] != token
                    or before['detection_name'] != after['detection_name']
                    or before['detection_score'] != after['detection_score']):
                raise ValueError('Source-index/class/score alignment differs')
            if after['detection_name'] not in classes:
                continue
            same_geometry = (np.allclose(after['translation'], before['translation'], atol=1e-6, rtol=0)
                             and np.allclose(after['size'], before['size'], atol=1e-6, rtol=0)
                             and np.allclose(after['rotation'], before['rotation'], atol=1e-6, rtol=0))
            if same_geometry:
                continue
            if not np.isfinite(before['translation'] + before['size'] + after['size']).all() or min(before['size'] + after['size']) <= 0:
                raise ValueError('Invalid box geometry')
            after['translation'][2] = before['translation'][2] - before['size'][2]/2 + after['size'][2]/2
            changed += 1
    return output, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--refined', required=True)
    parser.add_argument('--classes', default='motorcycle')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    classes = {c.strip() for c in args.classes.split(',') if c.strip()}
    if not classes or classes - {'car','truck','bus','trailer','pedestrian','motorcycle','bicycle'}:
        raise ValueError('Unsupported/empty anchor classes')
    baseline, refined, target = Path(args.baseline), Path(args.refined), Path(args.output)
    if target.resolve() in {baseline.resolve(), refined.resolve()} or target.exists():
        raise FileExistsError('Choose a NEW output file; preserve previous predictions')
    output, n = anchor_saved(json.loads(baseline.read_text()), json.loads(refined.read_text()), classes)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, allow_nan=False))
    report = {'policy': 'Class-specific detector bottom; keeps refined XY/yaw/size, replaces refined Z',
              'classes': sorted(classes), 'anchored_box_count': n,
              'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
              'refined_sha256': hashlib.sha256(refined.read_bytes()).hexdigest(),
              'limitations': ['Not a fitted ground plane.', 'No tracker/model rerun; legacy reverse-tracking observations remain.',
                             'Requires original source-index/score/class order, without gap filling.']}
    target.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print('Saved', target, '| anchored boxes:', n)


if __name__ == '__main__':
    main()
