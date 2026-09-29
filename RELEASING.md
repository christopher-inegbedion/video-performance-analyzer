# Releasing

## One-time setup

PyPI needs to be told that this repository is allowed to publish. This uses
Trusted Publishing, so there is no API token to create, store or leak.

1. Sign in at <https://pypi.org> (create an account if you have not).
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Under **Add a new pending publisher**, enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `video-performance-analyzer` |
   | Owner | `christopher-inegbedion` |
   | Repository name | `video-performance-analyzer` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

4. In the GitHub repo, go to **Settings → Environments → New environment** and
   create one called `pypi`. Adding yourself as a required reviewer there means
   every publish waits for your approval, which is worth it.

"Pending publisher" is correct for a project that does not exist on PyPI yet —
it becomes a normal publisher on the first successful upload.

## Each release

```bash
# 1. bump the version
#    pyproject.toml  ->  version = "0.2.0"
#    vpa/__init__.py ->  __version__ = "0.2.0"

# 2. commit and tag — the tag MUST match the version or CI fails the release
git commit -am "Release 0.2.0"
git tag v0.2.0
git push && git push --tags

# 3. create the release, which triggers the publish workflow
gh release create v0.2.0 --generate-notes
```

The workflow runs lint and the full test suite before building, so a release
cannot ship code that fails CI.

## Versioning

Semantic versioning. While at `0.x`, breaking changes bump the minor version.

Note that PyPI **never lets you reuse a version number**, even after deleting a
release. Test with a dry run before the first real publish:

```bash
python -m build
twine check dist/*
```

To rehearse the whole pipeline, use TestPyPI: add a `repository-url` of
`https://test.pypi.org/legacy/` to the publish step and register the same
trusted publisher at <https://test.pypi.org>.
