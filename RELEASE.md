# Making a release

Releases are managed with the `release.yml` GitHub Actions workflow. It is
started from the Actions tab; everything after that runs automatically or on
the GitHub website, and no local git commands are needed.

The workflow gates publication behind two human steps. It drafts a pull
request, a person merges it, a draft GitHub release is created, and nothing
reaches PyPI until a person publishes that release.

---

## Overview

| Step | Who does it | What happens |
|------|-------------|--------------|
| 1 | You (Actions tab) | Run **Release**, choosing the version bump |
| 2 | Workflow (automatic) | Version, changelog and citation committed to a `v*` branch |
| 3 | You (GitHub website) | Write the breaking changes, review, merge the draft PR |
| 4 | Workflow (automatic) | Draft GitHub release created from the changelog entry |
| 5 | You (GitHub website) | Publish the draft release |
| 6 | Workflow (automatic) | Package built and published to PyPI |

---

## Step 1 — Start the workflow

1. Go to **Actions → Release** in the repository.
2. Click **Run workflow** and fill in the three inputs.

### `bump`

Applied to the version the repository currently carries, which between
releases is a development version such as `1.4.0.dev0`.

| Choice | From `1.4.0.dev0` | From `1.4.0` | Use it for |
|--------|-------------------|--------------|------------|
| `stable` | `1.4.0` | `1.4.0` | the usual release, finalizing an open cycle |
| `patch` | `1.4.0` | `1.4.1` | a fix release |
| `minor` | `1.5.0` | `1.5.0` | skipping the open cycle's number |
| `major` | `2.0.0` | `2.0.0` | a release that breaks compatibility broadly |
| `rc` | `1.4.0rc0` | `1.4.0rc0` | a release candidate |
| `dev` | `1.4.0.dev1` | `1.5.0.dev0` | opening the next cycle after a release |

`stable` is what a release cut from an open development cycle needs.

### `repository`

`pypi` for a real release. `testpypi` uploads to TestPyPI instead and pushes
no branch, opens no pull request and creates no release, so nothing is burned
in git.

### `dry_run`

Bumps the version, builds the package and smoke tests it, then stops. Nothing
is pushed or published. Use it to check that the version resolves the way you
expect before running the release for real.

---

## Step 2 — Automated prep (no action needed)

The **Prepare release** job runs on `main` and:

1. Resolves the new version from the current one and the bump.
2. Updates `mf6adj/version.py`, and `CITATION.cff` to the version being
   released.
3. Generates the `changelog/CHANGELOG.md` entry from the pull requests merged
   since the last tag.
4. Builds the package, checks it with `twine`, and smoke tests both the wheel
   and the source distribution by importing them and asserting the version.
5. Pushes a `v<version>` branch and opens a **draft** pull request against
   `main`.

A release candidate stops before the pull request.

### Opening the next cycle

A `dev` bump takes a different path, because opening a cycle is not a release.
It writes no changelog entry, since nothing has been merged behind it, and
leaves `CITATION.cff` at the released version, which is the one to cite. Its
branch is named `open-<version>-cycle` rather than `v<version>`, because the
job that drafts a GitHub release triggers on a merged branch whose name starts
with `v`.

Run it once after each release, so a build from `main` is not mistaken for the
release just made. Merging its pull request is all it needs.

### Previewing the changelog

To see the entry before starting a release:

```bash
pixi run python scripts/generate_changelog.py --version 1.4.0 --dry-run
```

---

## Step 3 — Write the breaking changes, then merge

The generated entry has a `### Changes` section and nothing else. Two things
are added by hand on the release branch, editing
`changelog/CHANGELOG.md` directly on GitHub:

- **A `### Breaking changes` section**, above `### Changes`. This is the part
  of the release notes that is read most and the part no tool can write. A
  change belongs here if it stops something working, or if it changes a number
  a previous release reported: someone holding results from an earlier version
  needs to know whether to run them again, and how far out they were.
- **Removing the chores.** The generator lists every merged pull request, so
  the entry arrives carrying dependabot bumps, continuous integration changes
  and the development-cycle commit. Keep `feat`, `fix`, `refactor` and `build`;
  drop `ci`, `test`, `style` and `chore`, and the unprefixed dependabot titles.
  `docs` is usually a chore, but not when it adds a document that is itself a
  deliverable.

Then check `mf6adj/version.py`, mark the pull request **Ready for review**, and
merge it into `main`.

Merging triggers the **Draft release** job, which creates a draft GitHub
release tagged `v<version>`. Its notes are that version's changelog entry
alone, not the whole file.

---

## Step 4 — Publish the draft GitHub release

1. Go to the [Releases](https://github.com/INTERA-Inc/mf6adj/releases) page.
2. Open the draft and review the notes. This is the last chance to edit them.
3. Click **Publish release**.

A version number cannot be reused on PyPI. If something is wrong in the
package after this, the only remedy is another release.

---

## Step 5 — PyPI publish (automatic)

The **Publish package** job runs when the release is published. It builds with
`uv build`, validates with `twine check`, smoke tests the wheel and the source
distribution, and uploads to [PyPI](https://pypi.org/p/mf6adj) using trusted
publishing, so no API token is needed.

The job runs in the `release` GitHub environment. If that environment does not
exist, create it under **Settings → Environments** with the protection rules
you want.

---

## Step 6 — conda-forge

[regro's autotick bot](https://github.com/regro/cf-scripts) opens a pull
request on
[`mf6adj-feedstock`](https://github.com/conda-forge/mf6adj-feedstock) within a
few hours of the PyPI upload. It updates the version and the checksum and
nothing else, so **any dependency change since the last release has to be
added to its pull request by hand**, in `recipe/meta.yaml` under
`requirements: run:`.

Check it against `pyproject.toml`. A dependency that was removed is the one to
watch: the bot leaves it in the recipe, and conda users keep installing a
package the project no longer uses.

The recipe uses `{{ python_min }}`, which conda-forge pins globally, so the
minimum Python needs a change here only if this project's minimum is higher
than conda-forge's.

---

## Opening the next cycle

After the release is out, run the workflow again with `bump=dev` and merge the
pull request it opens. See step 2.

---

## Release checklist

- [ ] **Release** run from the Actions tab with the right bump
- [ ] Prep commit visible on the `v<version>` branch
- [ ] `### Breaking changes` written, chores removed from the entry
- [ ] `mf6adj/version.py` shows the right version
- [ ] Draft pull request merged into `main`
- [ ] Draft GitHub release reviewed and published
- [ ] **Publish package** job succeeded, and the version is on PyPI
- [ ] Dependency changes added to the conda-forge pull request
- [ ] Next cycle opened with `bump=dev`
