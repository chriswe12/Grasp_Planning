#!/usr/bin/env python3
"""Verify finite PPO checkpoint/optimizer/loss evidence for a catalog launch gate."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,default=ROOT,
                        help='Source/deployment root when checking pulled portable runs')
    parser.add_argument('--in-progress',action='store_true',
                        help='Check an active run; final cross-rank weight agreement remains unverified')
    args=parser.parse_args()
    args.project_root=args.project_root.resolve()
    cfg=json.loads((args.run/'run.json').read_text())
    catalog=Path(cfg['catalog'])
    for prefix in ('/workspace/project/', '/workspace/grasping_rl/'):
        if str(catalog).startswith(prefix):
            catalog=args.project_root/catalog.relative_to(prefix)
            break
    if not catalog.is_absolute():catalog=args.project_root/catalog
    digest=hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert digest==cfg['catalog_sha256'],'Catalog changed since training'
    checkpoints=[]
    for path in (args.run/'nn').glob('*.pth'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        checkpoints.append((int(checkpoint['epoch']),path,checkpoint))
    assert checkpoints,'No saved checkpoint'
    epoch,path,checkpoint=max(checkpoints,key=lambda x:x[0])
    assert epoch>=2,'Insufficient smoke epochs'
    def finite(x):
        if isinstance(x,torch.Tensor):
            assert torch.isfinite(x).all(),'Nonfinite checkpoint tensor'
            return 1
        if isinstance(x,dict):return sum(finite(v) for v in x.values())
        if isinstance(x,(list,tuple)):return sum(finite(v) for v in x)
        return 0
    count=finite(checkpoint)
    steps=[float(v['step']) for v in checkpoint['optimizer']['state'].values() if 'step' in v]
    assert steps and min(steps)>0,'No completed actor optimizer update'
    assert json.loads(path.with_suffix('.contract.json').read_text())==json.loads((args.run/'contract.json').read_text())
    events=EventAccumulator(str(args.run/'summaries')).Reload()
    losses={k:[x.value for x in events.Scalars(k)] for k in events.Tags()['scalars'] if k.startswith('losses/')}
    assert losses and all(math.isfinite(v) for values in losses.values() for v in values),'Missing/nonfinite loss metrics'
    report={'passed':True,'verification_scope':'in_progress' if args.in_progress else 'completed',
        'catalog_sha256':digest,'run':str(args.run),'checkpoint':str(path),
        'epochs':epoch,'transitions':checkpoint['frame'],'finite_tensors':count,
        'actor_optimizer_step_min':min(steps),'actor_optimizer_step_max':max(steps),'losses':losses}
    if cfg.get('multipart_geometry_assignment_version') == 1:
        import numpy as np
        contract=json.loads((args.run/'contract.json').read_text())
        assets=contract['object_assets']
        with np.load(catalog,allow_pickle=False) as data:
            selected=np.ones(len(data['target_ids']),dtype=bool) if cfg['catalog_split']=='all' else data['split']==cfg['catalog_split']
            expected_parts=set(data['target_part_indices'][selected].tolist())
        assigned=set()
        for rank in range(cfg.get('world_size',1)):
            record=json.loads((args.run/f'rank_{rank}.json').read_text())
            indices=record['assigned_part_indices']
            assert len(indices)==cfg['num_envs'], f'Rank {rank} lacks actual geometry assignment'
            assert set(indices)<=expected_parts, f'Rank {rank} contains a part outside the selected split'
            assert record['assigned_parts']==[assets[i]['part_key'] for i in indices]
            assigned.update(indices)
        assert assigned==expected_parts, 'Eligible training parts are missing from the active environments'
        report['geometry_assignment_verified']=True
        report['active_part_count']=len(assigned)
        report['active_parts']=sorted(assets[i]['part_key'] for i in assigned)
    if args.in_progress and cfg.get('world_size', 1) > 1:
        backends=[json.loads((args.run/f'backend_rank_{i}.json').read_text()) for i in range(cfg['world_size'])]
        assert all(b['rank']==i and b.get('gpu_uuid') for i,b in enumerate(backends)), 'Missing rank GPU identity'
        assert len({b['gpu_uuid'] for b in backends})==cfg['world_size'], 'Ranks share a physical GPU'
        rank_epochs=[]
        for i in range(cfg['world_size']):
            with (args.run/f'gpu_memory_rank_{i}.csv').open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            assert rows, f'Rank {i} has not completed an optimizer epoch'
            latest=int(rows[-1]['epoch'])
            assert latest>=epoch, f'Rank {i} has not reached checkpoint epoch {epoch}'
            rank_epochs.append(latest)
        report['active_ranks']=cfg['world_size']
        report['rank_epochs']=rank_epochs
        report['final_rank_weight_agreement']='pending completion'
    elif cfg.get('world_size', 1) > 1:
        ranks=[json.loads((args.run/f'training_rank_{i}.json').read_text()) for i in range(cfg['world_size'])]
        assert all(r['completed'] and r['rank']==i and r['world_size']==cfg['world_size']
                   for i,r in enumerate(ranks)), 'Incomplete distributed rank report'
        for key in ('epoch', 'actor_sha256', 'critic_sha256'):
            assert len({r.get(key) for r in ranks})==1, f'Distributed ranks disagree on {key}'
        report['synchronized_ranks']=cfg['world_size']
        report['actor_sha256']=ranks[0]['actor_sha256']
        report['critic_sha256']=ranks[0].get('critic_sha256')
    if cfg.get('resume_checkpoint'):
        baseline_path=Path(cfg['resume_checkpoint'])
        for prefix in ('/workspace/project/', '/workspace/grasping_rl/'):
            if str(baseline_path).startswith(prefix):
                baseline_path=args.project_root/baseline_path.relative_to(prefix)
                break
        if not baseline_path.is_absolute():baseline_path=args.project_root/baseline_path
        if (args.run/'resume_source.pth').is_file():
            baseline_path=args.run/'resume_source.pth'
        baseline=torch.load(baseline_path,map_location='cpu',weights_only=False)
        previous_steps=[float(v['step']) for v in baseline['optimizer']['state'].values() if 'step' in v]
        new_updates=min(steps)-max(previous_steps)
        assert new_updates>0,'Resume has not completed any new actor optimizer updates'
        assert checkpoint['frame']>baseline['frame'],'Resume has not collected new transitions'
        report['new_actor_optimizer_updates_min']=new_updates
        report['resume_checkpoint']=str(baseline_path)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='losses'},indent=2))


if __name__=='__main__':main()
