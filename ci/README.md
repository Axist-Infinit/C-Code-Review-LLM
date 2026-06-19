This CI workflow is parked at `ci/ci.yml` instead of `.github/workflows/ci.yml`
because it was pushed with a token lacking the GitHub `workflow` OAuth scope.

To enable it: rename this file to `.github/workflows/ci.yml` — either via the
GitHub web editor (Add file -> Create new file -> paste path), or locally with a
`workflow`-scoped token (`gh auth refresh -s workflow`), then commit.
