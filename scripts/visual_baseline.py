"""Capture FinSight screenshots at fixed viewports and compare them to a baseline.

    python scripts/visual_baseline.py capture   # write visual/current/*.png and report.json
    python scripts/visual_baseline.py update    # capture, then replace visual/baseline/
    python scripts/visual_baseline.py compare   # capture, then diff against visual/baseline/

Every mode exits 1 if a page shows horizontal overflow, a Streamlit exception or a console error;
`compare` also exits 1 when a screenshot differs from the baseline.

Screenshots depend on the operating system's fonts and rendering, so baselines are only
comparable on the platform that produced them (see visual/README.md).
"""
import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT / 'visual'
PORT = 8599
URL = f'http://localhost:{PORT}'

VIEWPORTS = {
    'desktop': {'width': 1440, 'height': 900},
    'tablet': {'width': 820, 'height': 1180},
    'mobile': {'width': 390, 'height': 844},
}
# (state name, sidebar page label, optional action run after the page loads)
STATES = [
    ('overview', 'Overview', None),
    ('evidence', 'Evidence analyst', None),
    ('evidence-answer', 'Evidence analyst', 'Model performance'),
    ('campaign', 'Campaign scenario', None),
    ('methodology', 'Methodology', None),
]
FREEZE = '*, *::before, *::after { animation: none !important; transition: none !important; caret-color: transparent !important; }'
MAX_DIFF_RATIO = 0.002  # share of pixels allowed to differ before `compare` fails


def port_in_use():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('127.0.0.1', PORT)) == 0


def start_server():
    if port_in_use():
        sys.exit(f'Port {PORT} is already in use; stop that server so the captured app is the one in this checkout.')
    cmd = [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', str(PORT),
           '--server.headless', 'true', '--client.toolbarMode', 'minimal',
           '--browser.gatherUsageStats', 'false']
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            if urllib.request.urlopen(f'{URL}/_stcore/health', timeout=1).status == 200:
                return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError('Streamlit did not become healthy within 60 seconds.')


def settle(page):
    page.wait_for_selector('[data-testid="stMain"] h1')
    page.wait_for_function('() => !document.querySelector("[data-testid=stStatusWidget]")', timeout=30000)
    page.add_style_tag(content=FREEZE)
    page.evaluate('() => document.fonts.ready')
    page.wait_for_timeout(400)


def set_sidebar(page, expanded):
    """Open or close the sidebar; on narrow screens it is collapsed by default and stays open after a click."""
    sidebar = page.locator('[data-testid="stSidebar"]')
    if (sidebar.get_attribute('aria-expanded') == 'true') == expanded:
        return
    page.locator('[data-testid="stExpandSidebarButton"]' if expanded else '[data-testid="stSidebarCollapseButton"]').first.click()
    page.wait_for_function(f'() => document.querySelector("[data-testid=stSidebar]").getAttribute("aria-expanded") === "{str(expanded).lower()}"')
    page.wait_for_timeout(400)


def capture_state(page, viewport, name, label, action):
    set_sidebar(page, True)
    page.locator('[data-testid="stSidebar"] [role="radiogroup"] label').filter(has_text=label).click()
    settle(page)
    if viewport['width'] < 768:  # below Streamlit's mobile breakpoint the sidebar is an overlay; close it
        set_sidebar(page, False)
    if action:
        page.get_by_role('button', name=action).click()
        settle(page)
    # Streamlit scrolls an inner container, so grow the viewport to the content height.
    height = page.evaluate('() => (document.querySelector("[data-testid=stMain]") || document.body).scrollHeight')
    page.set_viewport_size({'width': viewport['width'], 'height': max(viewport['height'], int(height))})
    page.mouse.move(viewport['width'] - 4, 4)  # park the pointer away from hover-only controls
    page.wait_for_timeout(300)
    metrics = page.evaluate('''() => ({
        horizontal_overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
        content_height: (document.querySelector("[data-testid=stMain]") || document.body).scrollHeight,
        exceptions: document.querySelectorAll('[data-testid="stException"]').length,
    })''')
    return metrics


def capture(out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    report = {}
    server = start_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for vp_name, viewport in VIEWPORTS.items():
                context = browser.new_context(viewport=viewport, device_scale_factor=1, locale='en-US',
                                              timezone_id='UTC', reduced_motion='reduce')
                page = context.new_page()
                console_errors = []
                page.on('console', lambda m: console_errors.append(m.text) if m.type == 'error' else None)
                started = time.time()
                page.goto(URL)
                settle(page)
                load_seconds = round(time.time() - started, 2)
                for name, label, action in STATES:
                    page.set_viewport_size(viewport)
                    metrics = capture_state(page, viewport, name, label, action)
                    metrics['load_seconds'] = load_seconds if name == 'overview' else None
                    page.screenshot(path=out_dir / f'{vp_name}-{name}.png')
                    report[f'{vp_name}-{name}'] = metrics
                report[f'{vp_name}-console_errors'] = console_errors
                context.close()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2))
    return report


def compare(baseline, current):
    shutil.rmtree(VISUAL / 'diff', ignore_errors=True)
    failures = []
    for base_file in sorted(baseline.glob('*.png')):
        cur_file = current / base_file.name
        if not cur_file.exists():
            failures.append(f'{base_file.name}: missing from current capture')
            continue
        a, b = Image.open(base_file).convert('RGB'), Image.open(cur_file).convert('RGB')
        if a.size != b.size:
            failures.append(f'{base_file.name}: size changed {a.size} -> {b.size}')
            continue
        diff = ImageChops.difference(a, b).convert('L').point(lambda v: 255 if v > 16 else 0)
        ratio = sum(1 for v in diff.getdata() if v) / (a.width * a.height)
        if ratio > MAX_DIFF_RATIO:
            (VISUAL / 'diff').mkdir(exist_ok=True)
            diff.save(VISUAL / 'diff' / base_file.name)
            failures.append(f'{base_file.name}: {ratio:.2%} of pixels differ')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['capture', 'update', 'compare'])
    mode = parser.parse_args().mode
    report = capture(VISUAL / 'current')
    problems = [k for k, v in report.items() if (isinstance(v, dict) and (v['horizontal_overflow'] or v['exceptions'])) or (isinstance(v, list) and v)]
    print(f'Captured {len(list((VISUAL / "current").glob("*.png")))} screenshots in {VISUAL / "current"}')
    if problems:
        print('Page problems (horizontal overflow, Streamlit exception, or console error):', ', '.join(problems))
    if mode == 'update' and problems:
        sys.exit('Baseline not updated: fix the problems above first.')
    if mode == 'update':
        if (VISUAL / 'baseline').exists():
            shutil.rmtree(VISUAL / 'baseline')
        shutil.copytree(VISUAL / 'current', VISUAL / 'baseline')
        print(f'Baseline updated: {VISUAL / "baseline"}')
    elif mode == 'compare':
        if not (VISUAL / 'baseline').exists():
            sys.exit('No baseline found. Run `update` first.')
        failures = compare(VISUAL / 'baseline', VISUAL / 'current')
        for line in failures:
            print('FAIL', line)
        sys.exit(1 if failures or problems else 0)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
