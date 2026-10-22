"""Render the real pytest JUnit results into an offline readable HTML report."""
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Create reports/tests.html from reports/pytest.xml; never invent test outcomes."""
    junit = ET.parse(ROOT / 'reports/pytest.xml').getroot()
    tests = list(junit.iter('testcase'))
    rows = []
    for test in tests:
        failure = test.find('failure')
        error = test.find('error')
        skipped = test.find('skipped')
        status = 'FAIL' if failure is not None else 'ERROR' if error is not None else 'SKIP' if skipped is not None else 'PASS'
        rows.append(f'<tr><td>{escape(test.attrib["name"])}</td><td>{status}</td><td>{escape(test.get("time", ""))}</td></tr>')
    failures = sum(1 for t in tests if t.find('failure') is not None or t.find('error') is not None)
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>MMA3001 test report</title>
<style>body{{max-width:1100px;margin:40px auto;padding:20px;font:16px/1.6 system-ui;color:#173045}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd6df;padding:10px;text-align:left}}th{{background:#e9f3f8}}</style>
<h1>MMA3001 offline test report</h1><p>{len(tests)} collected tests; {failures} failures/errors.</p>
<p>Generated from pytest JUnit XML. Tests cover hand-computable arithmetic, invalid pairs, duplicate handling,
chronological splits, train-only preprocessing, unknown categories, forbidden predictors, frozen validation/holdout
reconciliation, grouped error totals, telemetry integrals, weather look-ahead and benchmark helpers.
No statement of percentage code coverage or deployment reliability is implied.</p>
<p>Separate full-notebook reproduction evidence: <a href="reproduction/summary.json">summary</a>.
Repeated performance measurements: <a href="performance/validation_performance.csv">CSV</a>.
Re-run with <code>python -m pytest --junitxml=reports/pytest.xml</code>, then <code>python -m tools.build_reports</code>.</p>
<table><thead><tr><th>Test</th><th>Result</th><th>Seconds</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>'''
    (ROOT / 'reports/tests.html').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
