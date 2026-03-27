#!/usr/bin/env bash
set -euo pipefail

cd /workspace/chhat-project
source .venv/bin/activate

if [[ ! -f .env ]]; then
  echo ".env missing at /workspace/chhat-project/.env"
  exit 1
fi

python3 - <<'PY'
import os
import boto3
from dotenv import load_dotenv

load_dotenv('/workspace/chhat-project/.env')
target = '/workspace/chhat-project/backend/references'
os.makedirs(target, exist_ok=True)
if len([f for f in os.listdir(target) if os.path.isfile(os.path.join(target, f))]) >= 50:
    print('References already present; skipping download.')
    raise SystemExit(0)

client = boto3.client(
    's3',
    region_name=os.getenv('DO_SPACES_REGION'),
    endpoint_url=os.getenv('DO_SPACES_ENDPOINT'),
    aws_access_key_id=os.getenv('DO_SPACES_KEY'),
    aws_secret_access_key=os.getenv('DO_SPACES_SECRET'),
)
resp = client.list_objects_v2(Bucket=os.getenv('DO_SPACES_BUCKET', 'chhat'), Prefix='references/')
count = 0
for obj in resp.get('Contents', []):
    key = obj['Key']
    name = key.split('/')[-1]
    if not name:
        continue
    out = os.path.join(target, name)
    client.download_file(os.getenv('DO_SPACES_BUCKET', 'chhat'), key, out)
    count += 1
print(f'Downloaded {count} references')
PY

python brand_classifier.py --embed-batch-size 8
python train.py

echo "Training cycle complete. Artifacts: runs/ and backend/classifier_model/"
