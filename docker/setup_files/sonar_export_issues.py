#!/usr/bin/env python
"""Export all open issues, security hotspots, measures and the quality gate of a SonarQube project.

Usage: sonar-export-issues [PROJECT_KEY] [-o OUTDIR]
PROJECT_KEY defaults to sonar.projectKey in ./sonar-project.properties (i.e. the CI project).
Needs SONAR_HOST_URL and SONAR_TOKEN in the environment.
Exit code is 1 when the quality gate is ERROR.
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

import requests

PAGE_SIZE = 500
# /api/issues/search refuses to page past 10k results
MAX_RESULTS = 10000
SEVERITY_ORDER = ['BLOCKER', 'CRITICAL', 'MAJOR', 'HIGH', 'MEDIUM', 'MINOR', 'LOW', 'INFO']
MEASURE_KEYS = [
    ['ncloc', 'coverage', 'duplicated_lines_density', 'software_quality_reliability_issues',
     'software_quality_security_issues', 'software_quality_maintainability_issues'],
    ['ncloc', 'coverage', 'duplicated_lines_density', 'bugs', 'vulnerabilities', 'code_smells'],
]


def key_from_properties(path='sonar-project.properties'):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*sonar\.projectKey\s*[=:]\s*(\S+)', line)
            if m:
                return m.group(1)
    return None


class Sonar:

    def __init__(self, url, token):
        self.url = url.rstrip('/')
        self.session = requests.Session()
        self.session.headers['Authorization'] = f'Bearer {token}'

    def get(self, path, **params):
        resp = self.session.get(f'{self.url}{path}', params=params, timeout=60)
        if resp.status_code in (401, 403):
            sys.exit(f'ERROR: {resp.status_code} from {path} - check SONAR_TOKEN and its permissions')
        resp.raise_for_status()
        return resp.json()

    def paged(self, path, item_field, **params):
        items, page = [], 1
        while True:
            data = self.get(path, p=page, ps=PAGE_SIZE, **params)
            items.extend(data.get(item_field, []))
            total = data.get('paging', {}).get('total', data.get('total', 0))
            if page * PAGE_SIZE >= min(total, MAX_RESULTS):
                if total > MAX_RESULTS:
                    print(f'WARNING: {path} has {total} results; only the first {MAX_RESULTS} '
                          'can be exported via the API', file=sys.stderr)
                return items
            page += 1


def file_of(component, project_key):
    return component.removeprefix(f'{project_key}:')


def impacts_of(issue):
    return ';'.join(f"{i['softwareQuality']}:{i['severity']}" for i in issue.get('impacts', []))


def severity_rank(sev):
    return SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else len(SEVERITY_ORDER)


def fetch_measures(sonar, key):
    for keys in MEASURE_KEYS:
        try:
            data = sonar.get('/api/measures/component', component=key, metricKeys=','.join(keys))
        except requests.HTTPError:
            continue
        return {m['metric']: m.get('value') for m in data['component'].get('measures', [])}
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_key', nargs='?', default=key_from_properties())
    parser.add_argument('-o', '--out-dir', default='sonar-report')
    args = parser.parse_args()

    url, token = os.environ.get('SONAR_HOST_URL'), os.environ.get('SONAR_TOKEN')
    if not url or not token:
        sys.exit('ERROR: SONAR_HOST_URL and SONAR_TOKEN must be set')
    if not args.project_key:
        sys.exit('ERROR: no project key given and none found in ./sonar-project.properties')
    key = args.project_key
    sonar = Sonar(url, token)

    issues = sonar.paged('/api/issues/search', 'issues', componentKeys=key, resolved='false')
    hotspots = sonar.paged('/api/hotspots/search', 'hotspots', project=key, status='TO_REVIEW')
    gate = sonar.get('/api/qualitygates/project_status', projectKey=key)['projectStatus']
    measures = fetch_measures(sonar, key)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'issues.json'), 'w') as f:
        json.dump(issues, f, indent=2)
    with open(os.path.join(args.out_dir, 'hotspots.json'), 'w') as f:
        json.dump(hotspots, f, indent=2)
    with open(os.path.join(args.out_dir, 'issues.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['severity', 'type', 'impacts', 'rule', 'file', 'line', 'message', 'effort', 'key'])
        for i in sorted(issues, key=lambda i: (severity_rank(i.get('severity')), i['component'], i.get('line', 0))):
            writer.writerow([i.get('severity', ''), i.get('type', ''), impacts_of(i), i['rule'],
                             file_of(i['component'], key), i.get('line', ''), i['message'],
                             i.get('effort', ''), i['key']])

    by_severity = collections.Counter(i.get('severity', '?') for i in issues)
    by_file = collections.Counter(file_of(i['component'], key) for i in issues)
    lines = [f'Project:      {key}', f'Server:       {sonar.url}', f"Quality gate: {gate['status']}"]
    for c in gate.get('conditions', []):
        if c['status'] != 'OK':
            lines.append(f"  - {c['metricKey']}: {c.get('actualValue')} "
                         f"({c['comparator']} {c.get('errorThreshold')}) -> {c['status']}")
    lines.append('Measures:')
    lines += [f'  {k}: {v}' for k, v in measures.items()]
    lines.append(f'Open issues: {len(issues)}')
    lines += [f'  {s}: {n}' for s, n in sorted(by_severity.items(), key=lambda x: severity_rank(x[0]))]
    lines.append(f'Security hotspots to review: {len(hotspots)}')
    lines.append('Top files:')
    lines += [f'  {n:5d}  {path}' for path, n in by_file.most_common(15)]
    summary = '\n'.join(lines)
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')

    print(summary)
    print(f'\nWrote issues.json, issues.csv, hotspots.json, summary.txt to {args.out_dir}/')
    return 1 if gate['status'] == 'ERROR' else 0


if __name__ == '__main__':
    sys.exit(main())
