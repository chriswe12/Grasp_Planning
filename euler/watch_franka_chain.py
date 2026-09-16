#!/usr/bin/env python3
"""Watch every segment and pull checkpoints after each job, surviving terminal closure."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
manifest=Path(sys.argv[1]).resolve()
chain=json.loads(manifest.read_text())
results=manifest.parent/'euler_results'
env=dict(os.environ,EULER_CONFIG_PATH=chain['config'],EULER_LOCAL_RESULTS_DIR=str(results),PYTHONUNBUFFERED='1')
status=[]
for job in [chain['initial_job']]+[entry['job_id'] for entry in chain['jobs']]:
    print(f'[CHAIN] Watching and pulling job {job}',flush=True)
    result=subprocess.run(['bash',str(ROOT/'euler/watch_and_pull.sh'),str(job),'60'],cwd=ROOT,env=env)
    status.append(dict(job_id=job,watch_exit_code=result.returncode))
    manifest.with_suffix('.watch.json').write_text(json.dumps(dict(jobs=status,results=str(results)),indent=2)+'\n')
    if result.returncode:
        print(f'[CHAIN] Job {job} did not complete successfully. Logs were requested; dependent jobs cannot train.',flush=True)
validation=[]
for path in results.rglob('evaluation.json'):
    value=json.loads(path.read_text())
    if value.get('catalog_split')=='validation' and value.get('coverage_complete'):
        validation.append(dict(report=str(path),checkpoint=value['checkpoint'],macro_part_success=value['macro_part_success']))
summary=dict(jobs=status,results=str(results),validation=validation,
    best_validation=max(validation,key=lambda x:x['macro_part_success']) if validation else None)
manifest.with_suffix('.results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
sys.exit(0 if all(item['watch_exit_code']==0 for item in status) else 1)
