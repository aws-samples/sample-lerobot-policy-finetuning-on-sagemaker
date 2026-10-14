# Contributing Guidelines

Thank you for your interest in contributing to this project. Bug reports, fixes, new features and documentation
corrections are all welcome.

Please read this document before you open an issue or a pull request. It tells you what we need in order to respond well.


## Reporting Bugs/Feature Requests

Please use the GitHub issue tracker to report bugs or suggest features.

Before you file an issue, check the open and recently closed issues so that the same problem is not reported twice.
Include as much information as you can. These details help most:

* A reproducible test case or a series of steps
* The version of our code you are using
* Any changes you made that relate to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Pull requests are welcome. Before you send one, please check that:

1. You are working from the latest source on the *main* branch.
2. Nobody else has already fixed the problem. Look at the open and recently merged pull requests.
3. You have opened an issue to discuss any large change. This avoids wasted work on both sides.

To send us a pull request, please:

1. Fork the repository.
2. Make your change and keep it focused. If you also reformat unrelated code, your change becomes hard to review.
3. Run the laptop checks and make sure they pass. They need no AWS credentials, Docker or GPU:
   `pip install -r requirements-dev.txt`, then `bash scripts/scrub_check.sh`, `pytest` and
   `python3 -m pipeline.policies_doc --check`. CI runs the same checks plus a shell syntax check, a `detect-secrets` scan against the audited
   `.secrets.baseline` and `bandit`, a Python
   security scanner. The full list with healthy outputs is in [docs/TESTING.md](docs/TESTING.md), section 1.
   `docs/POLICIES.md` and the four marked blocks of `README.md` (licence disclaimer, capability matrix, scenario table,
   instance fallback tables) are generated. If you changed `trainer/policies/*.yaml` or `pipeline/policies_doc.py`, run
   `python3 -m pipeline.policies_doc` to regenerate them instead of editing them by hand; `--check` covers both files.
4. Adding an AWS call? Build the client with `pipeline.aws.client(sess, service)` (or `pipeline.solution.get_client`). A
   bare `boto3.client(...)` works and is simply never attributed to the solution; nothing warns about it. Code that runs
   inside a job has no repository to import from and reads the `USER_AGENT_STRING` environment variable instead, as
   `trainer/hub_stage.py` does. `tests/test_solution.py` checks the real outbound header; run it after touching any of
   this. What is reported and why: README, [Anonymous usage metrics](README.md#anonymous-usage-metrics).
5. Changed `trainer/`? `cdk deploy` rebuilds the training image when that folder or `image.tag` changes. To rebuild without
   a deploy, `python3 -m pipeline.build_image --region <region> --tag v1 --publish-ssm` runs the same CodeBuild project;
   a build for an unchanged tree re-tags the existing image. `./finetune.sh` waits for a running publishing build (one
   started by `cdk deploy` or with `--publish-ssm`) before it submits anything; plain dev builds are ignored.
6. Commit to your fork with clear commit messages.
7. Send us a pull request. In the description, say what changed, why, and which of the checks you ran.
8. Watch for CI failures reported on the pull request and stay in the conversation.

GitHub provides additional documentation on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Verifying a release candidate
Maintainers: before tagging or merging a release candidate, run
`bash scripts/verify_fresh_clone.sh --ref <git ref> --out <report.md>`. It repeats the CI checks on a fresh clone and
prints a Markdown report with every account id masked, so the report can be pasted into an issue. What it checks, its
flags, and what a healthy report looks like are in
[docs/TESTING.md](docs/TESTING.md#1-laptop-only-no-aws-credentials-no-docker-no-gpu), section 1.

## Finding contributions to work on
Existing issues are a good place to start. The repository uses GitHub's default labels. Issues tagged 'help wanted' are
the best entry point.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the FAQ link and the contact address.


## Security issue notifications
If you find a potential security issue in this project, please tell AWS/Amazon Security through the
[vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a
public GitHub issue.


## Licensing

See the [LICENSE](LICENSE) file for the project's licence (MIT-0). We will ask you to confirm the licensing of your
contribution.
