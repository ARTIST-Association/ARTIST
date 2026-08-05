# Releasing a new version of `ARTIST`

The current workflow for releasing a new version of `ARTIST` is as follows:
1. Make sure the main branch is up-to-date and contains the version of the software that it is to be released.
2. On the main branch, update the version number in `pyproject.toml`. We use semantic versioning.
3. On the main branch, update the version number in `docs/conf.py`. We use semantic versioning.
4. Rebase the ``release-test`` branch onto the current main branch.
5. Push the ``release-test`` branch. This triggers a GitHub :octocat: action that will publish `ARTIST` to TestPyPi and
automatically verifies that the TestPyPi version of ``ARTIST`` works as planned by running all tests.
7. If the TestPyPI release completed successfully, create a draft release on GitHub and generate the release notes. Use the generated release notes as a starting point, refine their wording as needed, and then add the finalized release notes to `CHANGELOG.md`.
9. Make GitHub :octocat: release from the current main, including the corresponding version tag.
11. This will trigger an automatic Zenodo archive of the repository. Once this archive is available, update the Zenodo badge in the README to the latest version.
12. Rebase the ``release`` branch onto current main branch.
13. Push release branch. This will trigger a GitHub :octocat: action publishing the new release on PyPI.
