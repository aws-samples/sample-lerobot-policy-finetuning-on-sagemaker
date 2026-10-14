# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Durable infrastructure for fine-tuning LeRobot policies as SageMaker training jobs.

One stack, all a fresh account needs before ./finetune.sh:

  - ECR repo          : the training image (built by the CodeBuild project below, pinned by digest in SSM)
  - S3 bucket         : datasets/, hub/ (staged base weights), jobs/ (checkpoints + outputs), build-context/,
                        build-artifacts/ -- holds every dataset, checkpoint and published model of your runs
  - SageMaker role    : the execution role every smoke / staging / training job assumes (scoped to this bucket,
                        this repo, CloudWatch Logs and one Secrets Manager secret: the Hugging Face token, <project>/hf_token by default)
  - CodeBuild project : builds trainer/ (shipped as an S3 asset) into the ECR repo -- no Docker on your machine,
                        no ~13 GB (compressed) image upload from a laptop; writes the digest-pinned image URI to SSM on success
  - kickoff           : one StartBuild call during `cdk deploy` (async), re-fired only when trainer/ changes
  - SSM parameters    : /<project>/{project,region,bucket,sagemaker_role_arn,ecr_repo_uri} written here;
                        /<project>/image_uri and /<project>/image_src_hash written by the build. The CodeBuild
                        project name (<project>-image-build) is a stack output (ImageBuildProject), not a parameter
  - teardown          : `cdk destroy` removes everything above, including the bucket contents, the images, the two SSM
                        parameters the build wrote, the default-named Hugging Face token secret (<project>/hf_token; a
                        custom secrets.hf_token is the user's and stays) and the helper Lambdas' logs (retain_data=False,
                        the default). retain_data=True (`cdk deploy -c retainData=true`) keeps the
                        bucket, the repository and the secret. Left in every case: the CloudWatch log group
                        /aws/sagemaker/TrainingJobs, shared by every SageMaker job in the account.

The shape follows the DreamZero sample (aws-samples/sample-dreamzero-finetuning-on-sagemaker) minus its EC2
serving profile, LoRA merge and evaluation jobs.
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    CfnResource,
    Duration,
    RemovalPolicy,
    Stack,
    aws_codebuild as codebuild,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    aws_ssm as ssm,
    custom_resources as cr,
)
from cdk_nag import NagSuppressions
from constructs import Construct

# AWS's public Deep Learning Containers registry: the FROM image of trainer/sagemaker/Dockerfile and the only
# external registry the image build may pull from.
DLC_ACCOUNT = "763104351884"
S3_ROOT = "sagemaker"  # every prefix the pipeline reads or writes sits under s3://<bucket>/sagemaker/
COMPUTE = {
    "LARGE": codebuild.ComputeType.LARGE,        # 8 vCPU, 15 GB RAM; enough for the training image (~26 GB on disk, ~13 GB compressed push) with no compile step
    "XLARGE": codebuild.ComputeType.X_LARGE,
    "2XLARGE": codebuild.ComputeType.X2_LARGE,   # what the DreamZero sample used for its flash-attn compile
}


class LeRobotFinetuneStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, project: str, image_tag: str, build_compute: str = "LARGE",
                 kickoff_build: bool = True, bucket_name: str | None = None, hf_token_secret_name: str | None = None,
                 retain_data: bool = False,
                 **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)
        if build_compute not in COMPUTE:
            raise ValueError(f"buildCompute must be one of {sorted(COMPUTE)}, got {build_compute!r}")
        region, account, partition = Stack.of(self).region, Stack.of(self).account, self.partition

        # --- ECR ---------------------------------------------------------------------------------------------
        # Tags stay MUTABLE so a rebuild can re-push the same tag and use it as its layer cache; jobs never
        # reference a tag, they pin the digest the build published to SSM. Encryption is the ECR default (AES-256).
        repo = ecr.Repository(
            self, "TrainingImageRepo",
            repository_name=f"{project}-sagemaker-training",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            # retain_data=False (default): `cdk destroy` empties and deletes the repository with the stack;
            # True (`cdk deploy -c retainData=true`) keeps it and its images
            empty_on_delete=not retain_data,
            removal_policy=RemovalPolicy.RETAIN if retain_data else RemovalPolicy.DESTROY,
            lifecycle_rules=[
                ecr.LifecycleRule(description="untagged layers expire", rule_priority=1,
                                  tag_status=ecr.TagStatus.UNTAGGED, max_image_age=Duration.days(7)),
                ecr.LifecycleRule(description="developer builds expire", rule_priority=2,
                                  tag_status=ecr.TagStatus.TAGGED, tag_prefix_list=["dev-"], max_image_age=Duration.days(7)),
                ecr.LifecycleRule(description="keep the last 20 source-hash tags", rule_priority=3,
                                  tag_status=ecr.TagStatus.TAGGED, tag_prefix_list=["src-"], max_image_count=20),
            ],
        )

        # --- S3 ----------------------------------------------------------------------------------------------
        bucket = s3.Bucket(
            self, "AssetsBucket",
            bucket_name=bucket_name,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,  # write-once checkpoints; versioning every retrain would only add cost
            # retain_data=False (default): `cdk destroy` deletes every object and the bucket through CDK's S3 auto-delete
            # custom resource; True keeps the bucket (datasets, checkpoints, models) after the stack is gone
            removal_policy=RemovalPolicy.RETAIN if retain_data else RemovalPolicy.DESTROY,
            auto_delete_objects=not retain_data,
            lifecycle_rules=[
                # a job killed mid-upload leaves billed, invisible multipart parts behind
                s3.LifecycleRule(id="AbortOrphanedMultipartUploads", abort_incomplete_multipart_upload_after=Duration.days(7)),
                s3.LifecycleRule(id="ExpireSmokeJobs", prefix=f"{S3_ROOT}/jobs/smoke/", expiration=Duration.days(14)),
                s3.LifecycleRule(id="ExpireBuildContexts", prefix=f"{S3_ROOT}/build-context/", expiration=Duration.days(30)),
                s3.LifecycleRule(id="ExpireBuildArtifacts", prefix=f"{S3_ROOT}/build-artifacts/", expiration=Duration.days(90)),
            ],
        )

        # --- SageMaker execution role ---------------------------------------------------------------------------
        # CDK-named (IAM is global; a fixed name would make the stack deployable once per account). Scoped
        # inline policy instead of AmazonSageMakerFullAccess: this bucket, this repo, CloudWatch Logs.
        # The trust policy admits SageMaker only when it acts for a SageMaker resource of this account in this region
        # (aws:SourceAccount and aws:SourceArn, the confused-deputy guard from the SageMaker developer guide): a job in
        # another account that names this role ARN is refused even if its caller holds iam:PassRole there
        # (docs/THREAT_MODEL.md T-32). Both jobs this stack runs are training jobs, so the ARN pattern covers them.
        sm_role = iam.Role(self, "SageMakerExecutionRole", assumed_by=iam.ServicePrincipal(
            "sagemaker.amazonaws.com",
            conditions={"StringEquals": {"aws:SourceAccount": account},
                        "ArnLike": {"aws:SourceArn": f"arn:{partition}:sagemaker:{region}:{account}:*"}},
        ))
        # read on the whole bucket (datasets/, hub/, restored checkpoints); put and delete only under the two prefixes the
        # jobs write: jobs/* (smoke and training checkpoint sync, output.tar.gz and model.tar.gz, the staging job's outputs
        # under jobs/hub/) and hub/* (the staging job's Hugging Face cache mirror and _STAGED.json). The checkpoint sync
        # agent also mirrors container-side deletions there (hub_stage.py removes blobs/ and .locks/). The dataset and build
        # prefixes are read-only for the job identity, so code running inside a job cannot poison a dataset or another
        # policy's staged weights (docs/THREAT_MODEL.md T-28); smoke/build-context prefixes expire through lifecycle rules.
        if retain_data:
            # a retained bucket keeps its TLS-only bucket policy too (CloudFormation would otherwise delete the policy
            # resource and leave the bucket without the aws:SecureTransport deny)
            bucket.policy.apply_removal_policy(RemovalPolicy.RETAIN)
        bucket.grant_read(sm_role)
        bucket.grant_put(sm_role, f"{S3_ROOT}/jobs/*")
        bucket.grant_put(sm_role, f"{S3_ROOT}/hub/*")
        bucket.grant_delete(sm_role, f"{S3_ROOT}/jobs/*")
        bucket.grant_delete(sm_role, f"{S3_ROOT}/hub/*")
        repo.grant_pull(sm_role)
        sm_role.add_to_policy(iam.PolicyStatement(
            sid="TrainingJobLogsAndMetrics",
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
            resources=[f"arn:{partition}:logs:{region}:{account}:log-group:/aws/sagemaker/*"],
        ))
        sm_role.add_to_policy(iam.PolicyStatement(
            sid="TrainingJobMetrics", actions=["cloudwatch:PutMetricData"], resources=["*"],
            conditions={"StringEquals": {"cloudwatch:namespace": "/aws/sagemaker/TrainingJobs"}},
        ))
        if hf_token_secret_name:
            # Read by the base-weight staging job only (the pipeline never passes the token to a training job).
            sm_role.add_to_policy(iam.PolicyStatement(
                sid="HubTokenSecret", actions=["secretsmanager:GetSecretValue"],
                resources=[f"arn:{partition}:secretsmanager:{region}:{account}:secret:{hf_token_secret_name}*"],
            ))

        # --- SSM: stack facts the pipeline resolves at runtime ---------------------------------------------------
        ssm_prefix = f"/{project}"
        for key, value in {
            "project": project, "region": region, "bucket": bucket.bucket_name,
            "sagemaker_role_arn": sm_role.role_arn, "ecr_repo_uri": repo.repository_uri,
        }.items():
            ssm.StringParameter(self, f"Param{key.title().replace('_', '')}",
                                parameter_name=f"{ssm_prefix}/{key}", string_value=value)
        image_param = f"{ssm_prefix}/image_uri"          # written by the build (digest-pinned), not by CloudFormation
        src_hash_param = f"{ssm_prefix}/image_src_hash"  # so a build never turns into stack drift

        # --- CodeBuild image factory -----------------------------------------------------------------------------
        trainer_src = s3_assets.Asset(
            self, "TrainerBuildContext",
            path=str(Path(__file__).resolve().parent.parent.parent / "trainer"),
            exclude=["__pycache__", "*.pyc", ".pytest_cache", "*.egg-info", "README*"],
        )
        build_logs = logs.LogGroup(self, "ImageBuildLogs", retention=logs.RetentionDays.ONE_MONTH,
                                   removal_policy=RemovalPolicy.DESTROY)
        env = {k: codebuild.BuildEnvironmentVariable(value=v) for k, v in {
            "ECR_REPO_URI": repo.repository_uri,
            "ECR_REPO_NAME": repo.repository_name,
            "DLC_ACCOUNT": DLC_ACCOUNT,
            "IMAGE_TAG": image_tag,             # overridden per build (dev-<sha>)
            "CACHE_FROM_TAG": "auto",           # newest pushed tag; set explicitly to pin the cache source
            "LOCK_MODE": "enforce",             # enforce: pip freeze must equal sagemaker/requirements.lock | generate
            "PUBLISH_SSM": "1",                 # 1: write the digest-pinned URI to SSM (deploy builds); 0: dev builds
            "SSM_IMAGE_PARAM": image_param,
            "SSM_SRC_HASH_PARAM": src_hash_param,
            "ARTIFACTS_BUCKET": bucket.bucket_name,
            "DOCKER_SRC_HASH": trainer_src.asset_hash,
        }.items()}
        build = codebuild.Project(
            self, "ImageFactory",
            project_name=f"{project}-image-build",
            description=f"Builds trainer/ into {project}-sagemaker-training:<IMAGE_TAG>; on success writes the digest-pinned URI to SSM {image_param}",
            source=codebuild.Source.s3(bucket=trainer_src.bucket, path=trainer_src.s3_object_key),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=COMPUTE[build_compute],
                privileged=True,  # docker-in-docker: required to build the image; the project builds only this repo's
                                  # trainer/ asset and pushes only to this stack's ECR repo (no cdk-nag rule covers it)
            ),
            environment_variables=env,
            timeout=Duration.hours(2),
            logging=codebuild.LoggingOptions(cloud_watch=codebuild.CloudWatchLoggingOptions(log_group=build_logs)),
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "pre_build": {"commands": [
                        'aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | docker login --username AWS --password-stdin "${ECR_REPO_URI%%/*}"',
                        'aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | docker login --username AWS --password-stdin "${DLC_ACCOUNT}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"',
                        "\n".join([
                            'set -e',
                            'df -h / | tail -1; docker system df',
                            # The FROM image must exist in THIS region's DLC registry; fail with a readable message.
                            # image reference after the registry host of the first FROM line (registry region is a build arg)
                            'DLC_REF=$(grep -m1 -E "^FROM " sagemaker/Dockerfile | sed -E "s#^FROM +[^/]+/##")',
                            'docker manifest inspect "${DLC_ACCOUNT}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${DLC_REF}" >/dev/null '
                            '|| { echo "FATAL: DLC base ${DLC_REF} not found in ${AWS_DEFAULT_REGION}: the FROM line of trainer/sagemaker/Dockerfile '
                            'pins an image that is not published in this region; see https://github.com/aws/deep-learning-containers/blob/master/available_images.md"; exit 1; }',
                        ]),
                    ]},
                    "build": {"commands": ["\n".join([
                        'set -e',
                        'if [ "$CACHE_FROM_TAG" = "auto" ]; then',
                        '  CACHE_FROM_TAG=$(aws ecr describe-images --repository-name "$ECR_REPO_NAME"'
                        ' --query "sort_by(imageDetails[?imageTags],&imagePushedAt)[-1].imageTags[0]" --output text 2>/dev/null || true)',
                        'fi',
                        'CACHE_ARGS=""',
                        'if [ -n "$CACHE_FROM_TAG" ] && [ "$CACHE_FROM_TAG" != "None" ]; then',
                        '  echo "cache source: $ECR_REPO_URI:$CACHE_FROM_TAG"; CACHE_ARGS="--cache-from $ECR_REPO_URI:$CACHE_FROM_TAG"',
                        'fi',
                        # content hash of the build context: identical trees from `cdk deploy` (CDK asset zip) and
                        # scripts/build_image.sh (its own zip) get the same src- tag, so the second build is skipped and the
                        # already-pushed digest is re-tagged instead of producing a second, differently-hashed image
                        'CONTENT_HASH=$(find . -type f -not -name requirements.lock.built -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)',
                        'SRC_TAG="src-$CONTENT_HASH"; echo "content hash $CONTENT_HASH (asset $DOCKER_SRC_HASH)"; rm -f .skip_build',
                        'if [ "$LOCK_MODE" = "enforce" ] && aws ecr describe-images --repository-name "$ECR_REPO_NAME" --image-ids imageTag="$SRC_TAG" >/dev/null 2>&1; then',
                        '  echo "image for this trainer tree already exists as $ECR_REPO_URI:$SRC_TAG: re-tagging as $IMAGE_TAG, no rebuild"',
                        '  MANIFEST=$(aws ecr batch-get-image --repository-name "$ECR_REPO_NAME" --image-ids imageTag="$SRC_TAG" --query "images[0].imageManifest" --output text)',
                        '  aws ecr put-image --repository-name "$ECR_REPO_NAME" --image-tag "$IMAGE_TAG" --image-manifest "$MANIFEST" >/dev/null 2>&1 || true',
                        '  echo 1 > .skip_build',
                        'else',
                        # -f: the Dockerfile lives under sagemaker/, the build context is the asset root (trainer/)
                        '  DOCKER_BUILDKIT=1 docker build -f sagemaker/Dockerfile --build-arg BUILDKIT_INLINE_CACHE=1'
                        ' --build-arg DLC_REGION="$AWS_DEFAULT_REGION" --build-arg LOCK_MODE="$LOCK_MODE"'
                        ' --label org.opencontainers.image.revision="$DOCKER_SRC_HASH" --label org.opencontainers.image.version="$CONTENT_HASH"'
                        ' $CACHE_ARGS -t "$ECR_REPO_URI:$IMAGE_TAG" -t "$ECR_REPO_URI:$SRC_TAG" .',
                        # the lock the build produced (generate mode) or verified (enforce mode)
                        '  docker run --rm --entrypoint cat "$ECR_REPO_URI:$IMAGE_TAG" /opt/ml/code/requirements.lock.built > requirements.lock.built || true',
                        'fi',
                    ])]},
                    "post_build": {"commands": ["\n".join([
                        'if [ "$CODEBUILD_BUILD_SUCCEEDING" != "1" ]; then echo "build failed: not pushing, not touching SSM"; exit 1; fi',
                        'set -e',
                        'if [ "$LOCK_MODE" = "generate" ]; then',
                        '  aws s3 cp requirements.lock.built "s3://$ARTIFACTS_BUCKET/sagemaker/build-artifacts/$CODEBUILD_BUILD_ID/requirements.lock"',
                        '  echo "generate mode: lock uploaded, image NOT pushed"; exit 0',
                        'fi',
                        'if [ -f .skip_build ]; then echo "re-tagged existing image; nothing to push"; else',
                        '  CONTENT_HASH=$(find . -type f -not -name requirements.lock.built -not -name .skip_build -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)',
                        '  SRC_TAG="src-$CONTENT_HASH"; docker push "$ECR_REPO_URI:$IMAGE_TAG"; docker push "$ECR_REPO_URI:$SRC_TAG"',
                        'fi',
                        'DIGEST=$(aws ecr describe-images --repository-name "$ECR_REPO_NAME" --image-ids imageTag="$IMAGE_TAG"'
                        ' --query "imageDetails[0].imageDigest" --output text)',
                        'echo "pushed $ECR_REPO_URI:$IMAGE_TAG ($DIGEST)"',
                        'if [ "$PUBLISH_SSM" = "1" ]; then',
                        '  aws ssm put-parameter --name "$SSM_IMAGE_PARAM" --type String --overwrite --value "$ECR_REPO_URI@$DIGEST"',
                        '  aws ssm put-parameter --name "$SSM_SRC_HASH_PARAM" --type String --overwrite --value "$DOCKER_SRC_HASH"',
                        '  echo "SSM $SSM_IMAGE_PARAM -> $ECR_REPO_URI@$DIGEST"',
                        'fi',
                    ])]},
                },
            }),
        )
        # No trainer_src.grant_read(build): codebuild.Source.s3 above already grants s3:GetObject* on the exact asset key
        # (plus GetBucket*/List* on the bucket); Asset.grant_read would add <assets-bucket>/*, every stack's assets.
        repo.grant_pull_push(build)
        build.add_to_role_policy(iam.PolicyStatement(sid="DescribeImagesForCacheAndDigest",
                                                     actions=["ecr:DescribeImages"], resources=[repo.repository_arn]))
        build.add_to_role_policy(iam.PolicyStatement(
            sid="DlcBaseImagePull",
            actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
            resources=[f"arn:{partition}:ecr:{region}:{DLC_ACCOUNT}:repository/*"],
        ))
        build.add_to_role_policy(iam.PolicyStatement(
            sid="PublishImageParams", actions=["ssm:PutParameter"],
            resources=[f"arn:{partition}:ssm:{region}:{account}:parameter{image_param}",
                       f"arn:{partition}:ssm:{region}:{account}:parameter{src_hash_param}"],
        ))
        # developer builds read a source zip from build-context/ (scripts/build_image.sh) and drop lock files
        bucket.grant_read(build, f"{S3_ROOT}/build-context/*")
        bucket.grant_put(build, f"{S3_ROOT}/build-artifacts/*")

        # --- helper Lambda log group ------------------------------------------------------------------------------
        # Every AwsCustomResource below shares one CDK singleton Lambda. Without this, Lambda creates its log group on the
        # first invocation outside CloudFormation and `cdk destroy` leaves it behind; a managed group is deleted with the
        # stack, after the custom resources (they depend on the function, the function on the group).
        cr_logs = logs.LogGroup(self, "CustomResourceLogs", retention=logs.RetentionDays.ONE_MONTH,
                                removal_policy=RemovalPolicy.DESTROY)

        # --- deploy-time kickoff (async; re-fires only when trainer/ or the tag changes) ---------------------------
        kickoff_build_id = "n/a (imageBuild=skip)"
        if kickoff_build:
            call = cr.AwsSdkCall(
                service="CodeBuild", action="startBuild",
                parameters={"projectName": build.project_name, "environmentVariablesOverride": [
                    {"name": "IMAGE_TAG", "value": image_tag, "type": "PLAINTEXT"},
                    {"name": "DOCKER_SRC_HASH", "value": trainer_src.asset_hash, "type": "PLAINTEXT"},
                    {"name": "PUBLISH_SSM", "value": "1", "type": "PLAINTEXT"},
                ]},
                physical_resource_id=cr.PhysicalResourceId.of(f"{project}-image-build-kickoff"),
                output_paths=["build.id"],
            )
            kickoff = cr.AwsCustomResource(
                self, "ImageBuildKickoff", resource_type="Custom::ImageBuildKickoff",
                on_create=call, on_update=call, install_latest_aws_sdk=False, log_group=cr_logs,
                policy=cr.AwsCustomResourcePolicy.from_statements([
                    iam.PolicyStatement(actions=["codebuild:StartBuild"], resources=[build.project_arn])]),
            )
            kickoff.node.add_dependency(build)
            kickoff_build_id = kickoff.get_response_field("build.id")

        # --- delete-time cleanup: what `cdk destroy` would otherwise leave behind ----------------------------------
        # Each resource does nothing at create/update time and makes one SDK call while the stack is being deleted.
        # 1. The two SSM parameters the image build writes outside CloudFormation (digest-pinned image URI, source hash).
        #    DeleteParameters reports missing names in its response instead of failing, so a stack whose build never ran
        #    still deletes cleanly.
        cleanups = [cr.AwsCustomResource(
            self, "ImageParamsCleanup", resource_type="Custom::ImageParamsCleanup", install_latest_aws_sdk=False,
            log_group=cr_logs,
            on_delete=cr.AwsSdkCall(service="SSM", action="deleteParameters",
                                    parameters={"Names": [image_param, src_hash_param]}),
            policy=cr.AwsCustomResourcePolicy.from_statements([iam.PolicyStatement(
                actions=["ssm:DeleteParameters"],
                resources=[f"arn:{partition}:ssm:{region}:{account}:parameter{image_param}",
                           f"arn:{partition}:ssm:{region}:{account}:parameter{src_hash_param}"])]),
        )]
        # 2. The Hugging Face token secret. `cdk deploy` / finetune.sh create it outside the stack under the project's
        #    default name <project>/hf_token; the pipeline's role may only read it. Only that default-named secret is the
        #    stack's to delete: a custom secrets.hf_token names a secret the user owns, and it is left alone. Deleted
        #    without a recovery window (a token is re-issued at huggingface.co in a minute; a lingering copy is the
        #    risk); a missing secret is not an error for a forced delete. The resource exists in both modes and carries
        #    RETAIN when retain_data is set: flipping the mode on a deployed stack then only changes its DeletionPolicy,
        #    and CloudFormation never sends the Delete event that would run the call (a resource dropped from the
        #    template would receive one).
        if hf_token_secret_name and hf_token_secret_name == f"{project}/hf_token":
            cleanups.append(cr.AwsCustomResource(
                self, "HubTokenSecretCleanup", resource_type="Custom::HubTokenSecretCleanup",
                install_latest_aws_sdk=False, log_group=cr_logs,
                removal_policy=RemovalPolicy.RETAIN if retain_data else RemovalPolicy.DESTROY,
                on_delete=cr.AwsSdkCall(service="SecretsManager", action="deleteSecret",
                                        parameters={"SecretId": hf_token_secret_name, "ForceDeleteWithoutRecovery": True}),
                policy=cr.AwsCustomResourcePolicy.from_statements([iam.PolicyStatement(
                    actions=["secretsmanager:DeleteSecret"],
                    resources=[f"arn:{partition}:secretsmanager:{region}:{account}:secret:{hf_token_secret_name}*"])]),
            ))
        if not retain_data:
            # 3. CDK's S3 auto-delete provider is a second helper Lambda. Point its logging at the managed log group above
            #    (a property override on the generated function), so its logs are deleted with the stack too instead of
            #    landing in an implicit /aws/lambda/<function> group that outlives it. The auto-delete resource, the
            #    function and the group are deleted in that order, so every log line lands before the group goes.
            provider = self.node.try_find_child("Custom::S3AutoDeleteObjectsCustomResourceProvider")
            provider_fn = provider.node.try_find_child("Handler") if provider is not None else None
            if isinstance(provider_fn, CfnResource):
                provider_fn.add_property_override("LoggingConfig.LogGroup", cr_logs.log_group_name)

        # --- outputs -------------------------------------------------------------------------------------------
        CfnOutput(self, "Project", value=project)
        CfnOutput(self, "Region", value=region)
        CfnOutput(self, "Account", value=account)
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "S3Root", value=f"s3://{bucket.bucket_name}/{S3_ROOT}")
        CfnOutput(self, "EcrRepoUri", value=repo.repository_uri)
        CfnOutput(self, "SageMakerRoleArn", value=sm_role.role_arn)
        CfnOutput(self, "ImageBuildProject", value=build.project_name)
        CfnOutput(self, "SsmParameterPrefix", value=ssm_prefix)
        CfnOutput(self, "KickoffBuildId", value=kickoff_build_id)

        # --- cdk-nag: deliberate deviations, suppressed on the resource that carries them -------------------------
        NagSuppressions.add_resource_suppressions(bucket, [{
            "id": "AwsSolutions-S1",
            "reason": "Server access logging off: the bucket holds datasets and write-once checkpoints read only by this "
                      "account's SageMaker jobs; enable it (and versioning) if your compliance baseline requires them.",
        }])
        NagSuppressions.add_resource_suppressions(sm_role, [{
            "id": "AwsSolutions-IAM5",
            "reason": "Object-level wildcards are scoped to this stack's single bucket ARN (jobs read and put under run-named "
                      "prefixes created at runtime); ecr:GetAuthorizationToken supports only Resource:*; PutMetricData is "
                      "restricted by condition to the SageMaker training namespace; /aws/sagemaker/* log groups are "
                      "created by the service per job.",
        }], apply_to_children=True)
        NagSuppressions.add_resource_suppressions(build, [
            {"id": "AwsSolutions-CB4", "reason": "No S3 build artifacts to encrypt with a CMK: the output is a container image in "
                                                   "ECR (AES-256 at rest); switch the repo to KMS if your org requires CMKs."},
            {"id": "AwsSolutions-IAM5", "reason": "Wildcards come from repo.grant_pull_push on this stack's own repo, the "
                                                    "action wildcards (s3:GetObject*, GetBucket*, List*) of the CodeBuild S3 source "
                                                    "on the trainer asset key and its bucket, the pull-only statement on AWS's public "
                                                    "DLC registry (repository names there are not enumerable), the build-context/ and "
                                                    "build-artifacts/ prefix grants, and the CDK-generated CloudWatch Logs grants for "
                                                    "this project."},
        ], apply_to_children=True)
        if not retain_data:  # CDK's S3 auto-delete provider (retain_data=False only): a CDK-managed singleton Lambda
            provider = self.node.try_find_child("Custom::S3AutoDeleteObjectsCustomResourceProvider")
            if provider is not None:
                NagSuppressions.add_resource_suppressions(provider, [
                    {"id": "AwsSolutions-IAM4", "reason": "CDK-managed S3 auto-delete provider; AWSLambdaBasicExecutionRole grants only "
                                                           "CloudWatch Logs write for its own log group.",
                     "applies_to": ["Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"]},
                    {"id": "AwsSolutions-IAM5", "reason": "The provider deletes every object of this stack's bucket on `cdk destroy`; "
                                                           "the wildcard is the bucket's own object ARN."},
                    {"id": "AwsSolutions-L1", "reason": "The provider runtime is chosen by aws-cdk-lib, not this stack; it runs only "
                                                         "while the stack is being deleted."},
                ], apply_to_children=True)
        for c in cleanups:
            NagSuppressions.add_resource_suppressions(c, [
                {"id": "AwsSolutions-IAM5", "reason": "Delete-time cleanup scoped to this stack's own names: the Secrets Manager "
                                                        "ARN needs the six-character suffix wildcard the service appends; the SSM "
                                                        "statement names its two parameters exactly."},
            ], apply_to_children=True)
        handler = self.node.try_find_child("AWS679f53fac002430cb0da5b7982bd2287")
        self._checkov_skips(bucket, repo, [build_logs, cr_logs], handler)
        if handler is not None:
            NagSuppressions.add_resource_suppressions(handler, [
                {"id": "AwsSolutions-IAM4", "reason": "CDK-managed AwsCustomResource handler; AWSLambdaBasicExecutionRole grants "
                                                       "only CloudWatch Logs write for its own log group.",
                 "applies_to": ["Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"]},
                {"id": "AwsSolutions-L1", "reason": "The handler runtime is chosen by aws-cdk-lib, not this stack; it runs once "
                                                     "at deploy time in-account to start a CodeBuild build and processes no input."},
            ], apply_to_children=True)

    def _checkov_skips(self, bucket, repo, log_groups, handler) -> None:
        """Checkov (CloudFormation) skips for the deviations this stack takes on purpose. Each carries the same reason as the
        cdk-nag suppression and the docs/SECURITY.md table; a fork that needs the control removes the skip and adds it."""
        def skip(res, items):
            cfn = res if isinstance(res, CfnResource) else getattr(res, "node", None) and res.node.default_child
            if cfn is not None:
                cfn.add_metadata("checkov", {"skip": [{"id": i, "comment": c} for i, c in items]})
        skip(bucket, [
            ("CKV_AWS_18", "No access-logging bucket: the stack is a single bucket sample; add a log bucket if your baseline needs it (docs/SECURITY.md)"),
            ("CKV_AWS_21", "Versioning off on purpose: checkpoint pruning deletes 16-30 GB objects; versions would retain them; deploy with -c retainData=true to keep the bucket past the stack"),
        ])
        skip(repo, [
            ("CKV_AWS_51", "Mutable tags are the CodeBuild layer cache; training jobs pin the image by digest from SSM, never by tag"),
            ("CKV_AWS_136", "AES-256 (ECR default) encryption; a customer-managed KMS key is left to the deploying account"),
        ])
        for lg in log_groups:
            skip(lg, [("CKV_AWS_158", "CloudWatch Logs default encryption for 30-day build and helper-Lambda logs; a KMS key is left to the deploying account")])
        lambda_reason = "CDK-managed helper Lambda (runs once at deploy or delete time inside the account, no inbound traffic, no VPC resources to reach)"
        for fn in ([handler] if handler is not None else []) + [c for c in self.node.find_all() if c.node.id == "Handler" and isinstance(c, CfnResource) and c.cfn_resource_type == "AWS::Lambda::Function"]:
            skip(fn, [("CKV_AWS_115", lambda_reason), ("CKV_AWS_116", lambda_reason), ("CKV_AWS_117", lambda_reason)])
