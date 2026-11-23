#!/usr/bin/env bash
# Run pytest (with coverage) + SonarScanner against a personal sandbox project, then export the issues.
#
# SonarQube Community Build has no branch analysis: scanning with the repo's real sonar.projectKey
# would overwrite the main-branch results that the CI quality gate uses. So by default this scans
# into "<projectKey>-dev-<user>" instead. Override with SONAR_DEV_PROJECT_KEY.
#
# Usage: sonar-dev-scan [--no-tests] [extra sonar-scanner args, e.g. -X or -Dsonar.exclusions=...]
# Env:   SONAR_HOST_URL, SONAR_TOKEN (required), SONAR_DEV_PROJECT_KEY, PYTEST_ARGS, SONAR_REPORT_DIR

set -uo pipefail

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

RUN_TESTS=1
SCANNER_ARGS=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --no-tests) RUN_TESTS=0 ;;
        *) SCANNER_ARGS+=("$arg") ;;
    esac
done

: "${SONAR_HOST_URL:?SONAR_HOST_URL is not set (e.g. export SONAR_HOST_URL=https://sonarqube.example.dso)}"
: "${SONAR_TOKEN:?SONAR_TOKEN is not set (SonarQube > My Account > Security > Generate Token)}"

PROPS=sonar-project.properties
if [[ ! -f "$PROPS" ]]; then
    echo "ERROR: $PROPS not found in $(pwd). Run this from the repo root." >&2
    exit 2
fi

BASE_KEY=$(sed -n 's/^[[:space:]]*sonar\.projectKey[[:space:]]*[=:][[:space:]]*//p' "$PROPS" | tr -d '\r' | head -1)
if [[ -z "$BASE_KEY" ]]; then
    echo "ERROR: sonar.projectKey not found in $PROPS" >&2
    exit 2
fi
KEY="${SONAR_DEV_PROJECT_KEY:-${BASE_KEY}-dev-${USER:-$(whoami)}}"
if [[ "$KEY" == "$BASE_KEY" ]]; then
    echo "WARNING: scanning into the real project '$KEY' - this replaces the CI analysis on the server." >&2
fi
echo "==> Project key: $KEY  (server: $SONAR_HOST_URL)"

EXTRA=()
if [[ "$RUN_TESTS" == 1 ]]; then
    echo "==> Running pytest with coverage"
    # shellcheck disable=SC2086
    pytest --cov=. --cov-report=xml:coverage.xml --junitxml=pytest-report.xml ${PYTEST_ARGS:-}
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "WARNING: pytest exited with code $rc (5 = no tests collected); continuing with the scan." >&2
    fi
    [[ -f coverage.xml ]] && EXTRA+=("-Dsonar.python.coverage.reportPaths=coverage.xml")
    [[ -f pytest-report.xml ]] && EXTRA+=("-Dsonar.python.xunit.reportPath=pytest-report.xml")
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)
if [[ -n "$PY_VERSION" ]] && ! grep -q '^[[:space:]]*sonar\.python\.version' "$PROPS"; then
    EXTRA+=("-Dsonar.python.version=$PY_VERSION")
fi

echo "==> Running sonar-scanner"
sonar-scanner \
    -Dsonar.host.url="$SONAR_HOST_URL" \
    -Dsonar.token="$SONAR_TOKEN" \
    -Dsonar.projectKey="$KEY" \
    -Dsonar.projectName="$KEY" \
    -Dsonar.qualitygate.wait=true \
    "${EXTRA[@]}" \
    "${SCANNER_ARGS[@]}"
scan_rc=$?
if [[ $scan_rc -ne 0 ]]; then
    echo "WARNING: sonar-scanner exited with code $scan_rc (quality gate failed or analysis error); exporting anyway." >&2
fi

echo "==> Exporting issues"
sonar-export-issues "$KEY" -o "${SONAR_REPORT_DIR:-sonar-report}"
