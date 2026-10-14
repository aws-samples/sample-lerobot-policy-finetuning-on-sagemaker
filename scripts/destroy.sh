#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Remove everything `cdk deploy` created for this project, including an S3 bucket and ECR repository that were RETAINED
# (a stack deployed with `cdk deploy -c retainData=true`). Run it INSTEAD of `cdk destroy` while the stack still exists:
# it empties and deletes the bucket, deletes the repository, runs `cdk destroy`, then removes the /<project>/* SSM
# parameters. If the stack is already gone, pass --bucket <name> and it removes the retained bucket, the repository and
# the parameters. With the shipped default (retainData unset) a plain `cdk destroy` already removes
# everything, and this script is only a belt-and-braces path.
#   scripts/destroy.sh --region <region> [--profile <profile>] [--project <name>] [--bucket <name>] [--dry-run] [--yes] [--delete-secret]
# --delete-secret also removes the Hugging Face token secret (<project>/hf_token or secrets.hf_token; no recovery window).
# Training-job logs stay in the shared CloudWatch log group /aws/sagemaker/TrainingJobs; the script prints what is left.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REGION=""; PROFILE=""; PROJECT=""; BUCKET_ARG=""; DRY=0; YES=0; DELSECRET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2;; --profile) PROFILE="$2"; shift 2;; --project) PROJECT="$2"; shift 2;;
    --bucket) BUCKET_ARG="$2"; shift 2;;
    --dry-run) DRY=1; shift;; --yes) YES=1; shift;; --delete-secret) DELSECRET=1; shift;;
    -h|--help) awk 'NR>3 && !/^#/ {exit} NR>3 {sub(/^# ?/, ""); print}' "$0"; exit 0;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[[ -n "$REGION" ]] || REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-}}"
[[ -n "$REGION" ]] || { echo "FATAL: --region (or AWS_DEFAULT_REGION) is required" >&2; exit 2; }
[[ -n "$PROJECT" ]] || PROJECT=$(python3 -c "import json; print(json.load(open('project_config.json'))['project'])")
SECRET=$(python3 -c "import json; c=json.load(open('project_config.json')); print((c.get('secrets') or {}).get('hf_token') or '$PROJECT/hf_token')")
AWS=(aws --region "$REGION"); [[ -n "$PROFILE" ]] && AWS+=(--profile "$PROFILE")
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"; [[ -n "$PROFILE" ]] && export AWS_PROFILE="$PROFILE"
STACK="$PROJECT-infra"
# Rule: the region is passed to the CDK app explicitly (-c region below). The app otherwise takes its region from
# the ambient chain, so `cdk destroy` could target another region than --region names; this script also refuses a
# stack or a bucket that lives outside the --region it was given.
HAVE_STACK=1
"${AWS[@]}" cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1 || HAVE_STACK=0
if [[ "$HAVE_STACK" == "1" ]]; then
  BUCKET=$("${AWS[@]}" ssm get-parameter --name "/$PROJECT/bucket" --query Parameter.Value --output text 2>/dev/null || true)
  [[ -n "$BUCKET_ARG" && -n "$BUCKET" && "$BUCKET_ARG" != "$BUCKET" ]] && { echo "FATAL: --bucket $BUCKET_ARG differs from the stack's bucket $BUCKET; refusing"; exit 1; }
  [[ -n "$BUCKET" ]] || BUCKET="$BUCKET_ARG"
  REPO_URI=$("${AWS[@]}" ssm get-parameter --name "/$PROJECT/ecr_repo_uri" --query Parameter.Value --output text 2>/dev/null || true)
  REPO_NAME="${REPO_URI##*/}"; [[ -n "$REPO_NAME" ]] || REPO_NAME="$PROJECT-sagemaker-training"
else
  echo "no stack $STACK in $REGION"
  if [[ -z "$BUCKET_ARG" ]]; then
    echo "The stack is already gone. To remove what a retained deployment left behind, pass --bucket <name> (the bucket a"
    echo "  cdk deploy printed as BucketName; find it with: aws s3 ls | grep $PROJECT-infra-assetsbucket), or run by hand:"
    echo "    aws s3 rm s3://<bucket> --recursive --region $REGION && aws s3 rb s3://<bucket> --region $REGION"
    echo "    aws ecr delete-repository --repository-name $PROJECT-sagemaker-training --force --region $REGION"
    echo "    aws secretsmanager delete-secret --secret-id $SECRET --force-delete-without-recovery --region $REGION"
    exit 1
  fi
  BUCKET="$BUCKET_ARG"; REPO_NAME="$PROJECT-sagemaker-training"
fi
if [[ -n "$BUCKET" ]]; then
  LOC=$("${AWS[@]}" s3api get-bucket-location --bucket "$BUCKET" --query LocationConstraint --output text 2>/dev/null || true)
  [[ "$LOC" == "None" ]] && LOC="us-east-1"
  [[ -z "$LOC" || "$LOC" == "$REGION" ]] || { echo "FATAL: bucket $BUCKET lives in $LOC, not $REGION; refusing"; exit 1; }
fi
echo "project $PROJECT, region $REGION, stack $STACK$([[ "$HAVE_STACK" == "1" ]] || echo ' (absent)')"
if [[ -n "$BUCKET" ]]; then
  SIZE=$("${AWS[@]}" s3 ls "s3://$BUCKET" --recursive --summarize 2>/dev/null | awk '/Total Size/ {printf "%.1f GB", $3/1e9} /Total Objects/ {o=$3} END {printf " in %s objects\n", o}' || echo "size unknown")
  echo "bucket     s3://$BUCKET  ($SIZE)"
else
  echo "bucket     not found in SSM (/$PROJECT/bucket): pass --bucket <name> if a retained bucket exists; otherwise only the stack/SSM cleanup runs"
fi
IMAGES=$("${AWS[@]}" ecr describe-images --repository-name "$REPO_NAME" --query 'length(imageDetails)' --output text 2>/dev/null || echo 0)
echo "repository $REPO_NAME  ($IMAGES image digests)"
[[ "$HAVE_STACK" == "1" ]] && echo "stack      $("${AWS[@]}" cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo 'not found')"
echo "ssm        $("${AWS[@]}" ssm get-parameters-by-path --path "/$PROJECT/" --query 'length(Parameters)' --output text 2>/dev/null || echo 0) parameters under /$PROJECT/"
HAVE_SECRET=0; "${AWS[@]}" secretsmanager describe-secret --secret-id "$SECRET" >/dev/null 2>&1 && HAVE_SECRET=1
[[ "$HAVE_SECRET" == "1" ]] && echo "secret     $SECRET $([[ "$DELSECRET" == "1" ]] && echo '(will be deleted, --delete-secret)' || echo '(kept; pass --delete-secret to remove it)')"
if [[ "$DRY" == "1" ]]; then echo "(dry run: nothing deleted)"; exit 0; fi
if [[ "$YES" != "1" ]]; then
  read -r -p "This deletes every object above, irreversibly. Type the bucket name (${BUCKET:-none}) to continue: " CONFIRM
  [[ "$CONFIRM" == "${BUCKET:-none}" ]] || { echo "aborted"; exit 1; }
fi
if [[ -n "$BUCKET" ]]; then
  echo "== emptying s3://$BUCKET"; "${AWS[@]}" s3 rm "s3://$BUCKET" --recursive --only-show-errors
  echo "== deleting the bucket"; "${AWS[@]}" s3 rb "s3://$BUCKET" || echo "WARNING: bucket delete failed (the stack may own it: cdk destroy will retry)"
fi
echo "== deleting repository $REPO_NAME"; "${AWS[@]}" ecr delete-repository --repository-name "$REPO_NAME" --force >/dev/null 2>&1 || echo "  (no repository)"
if [[ "$HAVE_STACK" == "1" ]]; then
  echo "== cdk destroy $STACK"
  ( cd cdk && if [[ -f .venv/bin/activate ]]; then . .venv/bin/activate; fi && cdk destroy --force -c "region=$REGION" -c hfTokenPrompt=false "$STACK" )
fi
echo "== deleting SSM parameters under /$PROJECT/"
for n in $("${AWS[@]}" ssm get-parameters-by-path --path "/$PROJECT/" --query 'Parameters[].Name' --output text 2>/dev/null); do "${AWS[@]}" ssm delete-parameter --name "$n"; done
if [[ "$DELSECRET" == "1" && "$HAVE_SECRET" == "1" ]]; then
  echo "== deleting Secrets Manager secret $SECRET (no recovery window)"
  if ! OUT=$("${AWS[@]}" secretsmanager delete-secret --secret-id "$SECRET" --force-delete-without-recovery 2>&1 >/dev/null); then
    case "$OUT" in *ResourceNotFoundException*) echo "  (no secret)";; *) echo "WARNING: could not delete $SECRET: $OUT";; esac
  fi
fi
echo "done. Left behind:"
echo "  - CloudWatch log group /aws/sagemaker/TrainingJobs (shared with every SageMaker job in the account);"
echo "    aws logs put-retention-policy --log-group-name /aws/sagemaker/TrainingJobs --retention-in-days 30 --region $REGION"
if [[ "$HAVE_SECRET" == "1" && "$DELSECRET" != "1" ]]; then
  echo "  - Secrets Manager secret $SECRET:"
  echo "    aws secretsmanager delete-secret --secret-id $SECRET --force-delete-without-recovery --region $REGION"
fi
echo "  - the CDK bootstrap stack (CDKToolkit) and its assets bucket, shared by every CDK app in the account and region"
echo "  - on this machine: datasets/, runs/ and .finetune/"
