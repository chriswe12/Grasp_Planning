#!/usr/bin/env python3
"""Start one detached local GPU Fabrica training run after recorded readiness checks."""
import argparse
import fcntl
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=ROOT/'isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz')
    parser.add_argument('--controller-report',type=Path,default=ROOT/'artifacts/franka_randomized/controller_check.json')
    parser.add_argument('--smoke-report',type=Path,default=ROOT/'artifacts/franka_scaling/resume_probe.verification.json')
    parser.add_argument('--appearance-report',type=Path,default=ROOT/'artifacts/franka_scaling/batched_appearance_check.json')
    parser.add_argument('--iterations',type=int,default=2000)
    parser.add_argument('--num-envs',type=int,default=32,help='Environments per GPU; 32 was fastest in the local pencil-lab benchmark')
    parser.add_argument('--gpu-count',type=int,default=0,help='0 uses all installed GPUs; environments are per GPU')
    parser.add_argument('--checkpoint',type=Path,help='Resume a matching saved checkpoint')
    parser.add_argument('--cudnn-mode',choices=['compatible','native','disabled'],default='compatible')
    args=parser.parse_args()
    if args.iterations<1 or args.num_envs<1:parser.error('Positive iterations and environment count required')
    gpu_ids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
    gpu_count=args.gpu_count or len(gpu_ids)
    if not 1 <= gpu_count <= len(gpu_ids):parser.error(f'Only {len(gpu_ids)} local GPUs available')
    catalog=args.catalog.resolve();relative=catalog.relative_to(ROOT)
    digest=hashlib.sha256(catalog.read_bytes()).hexdigest()
    reports=[args.controller_report,args.smoke_report]
    with np.load(catalog,allow_pickle=False) as data:
        contract=json.loads(str(data['contract_json'].item()))
        if contract.get('appearance_randomization'):
            reports.append(args.appearance_report)
    checkpoint=''
    if args.checkpoint:
        source=args.checkpoint.resolve()
        if not source.is_file() or json.loads(source.with_suffix('.contract.json').read_text())!=contract:
            raise ValueError('Resume checkpoint must exist and match the catalog contract')
        checkpoint='/workspace/project/'+source.relative_to(ROOT).as_posix()
    for path in reports:
        report=json.loads(path.read_text())
        if not report.get('passed') or report.get('catalog_sha256')!=digest:
            raise ValueError(f'Current catalog needs a passing readiness report: {path}')
    existing=subprocess.check_output(['docker','ps','-q','--filter','label=codex.task=franka-fabrica-zed-training'],text=True).strip()
    if existing:
        print(f'A Franka/Fabrica training container is already running: {existing}')
        return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    name=f'franka-fabrica-zed-{stamp}'
    output=ROOT/'logs/franka_fabrica'/stamp
    output.mkdir(parents=True,exist_ok=False)
    inside='/workspace/project/'+output.relative_to(ROOT).as_posix()
    if args.checkpoint:
        snapshot=output/'resume_source.pth'
        shutil.copyfile(source,snapshot)
        shutil.copyfile(source.with_suffix('.contract.json'),snapshot.with_suffix('.contract.json'))
        checkpoint=inside+'/resume_source.pth'
    command=['docker','run','-d','--name',name,'--label','codex.task=franka-fabrica-zed-training',
        '--gpus',str(gpu_count),'--network','host','--shm-size','4g','--cpus',str(12*gpu_count),
        '--ulimit','nofile=65536:65536',
        '-e','ACCEPT_EULA=Y','-e','PRIVACY_CONSENT=Y','-e','OMNI_KIT_ALLOW_ROOT=1','-e',f'FRANKA_CUDNN_MODE={args.cudnn_mode}',
        '-v',f'{ROOT}:/workspace/project','-w','/workspace/project']
    cache=Path.home()/'.cache/torch/hub/checkpoints'
    if cache.exists():command+=['-v',f'{cache}:/root/.cache/torch/hub/checkpoints:ro']
    command+=['--entrypoint','/bin/bash','isaac-lab-euler:2.3.2',
        '/workspace/project/scripts/run_franka_fabrica_training.sh',
        '/workspace/project/'+relative.as_posix(),inside,str(args.num_envs),str(args.iterations),str(gpu_count),checkpoint]
    identifier=subprocess.check_output(command,text=True).strip()
    launch={'container':name,'container_id':identifier,'catalog':str(catalog),'catalog_sha256':digest,
            'output_directory':str(output),'num_envs':args.num_envs,'iterations':args.iterations,
            'gpu_count':gpu_count,'total_envs':args.num_envs*gpu_count,'checkpoint':str(args.checkpoint) if args.checkpoint else None,
            'resume_snapshot':str(output/'resume_source.pth') if args.checkpoint else None,
            'cudnn_mode':args.cudnn_mode,
            'command':command,'started_at_local':stamp,'automatic_post_training_evaluation':['validation','test']}
    (output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    (ROOT/'artifacts/franka_fabrica/latest_launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch,indent=2))


if __name__=='__main__':
    lock_path=ROOT/'artifacts/franka_fabrica/launch.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
