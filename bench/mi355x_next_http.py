"""Matched HTTP comparison of previous INT8, next INT8 and native rotated FP4."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

from run_http import Client


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--image', default='startlux-prism:mi355x-next')
    p.add_argument('--out', required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--reps', type=int, default=100)
    p.add_argument('--port', type=int, default=18122)
    p.add_argument('--profiles', default='int8-stock,int8-next,mxfp4-r32')
    p.add_argument('--check-concurrency', action='store_true')
    a = p.parse_args()
    workspace, output = Path(a.workspace).resolve(), Path(a.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bodies = json.loads(Path(a.manifest).read_text())['requests']
    folder = '/workspace/models/StartLux-Decision-v1.1-27B-AutoThink'
    gguf = folder + '/StartLux-Decision-v1.1-27B-AutoThink-Q8_0.gguf'
    artifact = '/workspace/models/StartLux-Decision-v1.1-27B-AutoThink-MXFP4-R32'
    calib = '/workspace/StartLux-Prism-mi355x/results/mi355x-27b-calib.pt'
    report = dict(image=a.image, rounds=a.rounds, measurements=[], health={}, responses={})
    profiles = a.profiles.split(',')
    for round_no in range(a.rounds):
        order = profiles if round_no % 2 == 0 else list(reversed(profiles))
        for profile in order:
            fp4 = profile == 'mxfp4-r32'
            name = 'startlux-prism-mi355x-next-http'
            command = ['docker', 'run', '--rm', '--name', name, '--network', 'host', '--ipc=host',
                       '--device=/dev/kfd', '--device=/dev/dri', '--group-add', 'video',
                       '--security-opt', 'seccomp=unconfined', '-e', 'HIP_VISIBLE_DEVICES=' + os.getenv('HIP_VISIBLE_DEVICES', '0'),
                       '-e', 'OMP_NUM_THREADS=8', '-v', str(workspace) + ':/workspace:ro',
                       '-v', 'startlux-prism-mi355x-next-triton:/root/.triton']
            if profile == 'int8-stock':
                for key, value in dict(INTRA_WARPS=8, HREC_BV=16, OUT_BV=32, HREC_NS=1, OUT_NS=1).items():
                    command += ['-e', f'STARTLUX_PRISM_{key}={value}']
            command += [a.image, '--gguf', folder if fp4 else gguf, '--model-dir', folder,
                        '--precision', 'mxfp4' if fp4 else 'int8', '--calib', calib,
                        '--port', str(a.port), '--max-tokens', '4096', '--graphs', '--graph-max-tokens', '1536',
                        '--prefix-cache', '0', '--mmproj', 'none', '--think-mode', 'never',
                        '--max-think-budget', '64', '--think-seqs', '1', '--think-kv-rows', '4096']
            if fp4:
                command += ['--mxfp4-artifact', artifact]
            log_path = output.with_name(f'http-{round_no}-{profile}.log')
            with log_path.open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    base = 'http://127.0.0.1:' + str(a.port)
                    deadline = time.monotonic() + 900
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f'{profile} exited: {log_path}')
                        try:
                            health = json.load(urllib.request.urlopen(base + '/health', timeout=2))
                            break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError('server startup')
                            time.sleep(.25)
                    report['health'][profile] = health
                    print('READY', round_no, profile, flush=True)
                    client = Client(base + '/v1/systemone')
                    for suite in ('yesno', 'review1', 'readme3', 'startlux3', 'longpage2'):
                        body = json.dumps(bodies[suite]['body']).encode()
                        for _ in range(5):
                            _, reference = client.call(body)
                        stable = reference['answers'], reference['usage']
                        samples = []
                        for _ in range(a.reps):
                            ms, got = client.call(body)
                            assert (got['answers'], got['usage']) == stable, (profile, suite, 'unstable response')
                            samples.append(ms)
                        old = report['responses'].setdefault(profile, {}).setdefault(suite, reference)
                        assert (old['answers'], old['usage']) == stable, (profile, suite, 'cross-round drift')
                        if profile == 'int8-next' and suite in report['responses'].get('int8-stock', {}):
                            prior = report['responses']['int8-stock'][suite]
                            assert (prior['answers'], prior['usage']) == stable, ('INT8 API output changed', suite)
                        row = dict(round=round_no, profile=profile, suite=suite, samples_ms=samples,
                                   median_ms=statistics.median(samples))
                        report['measurements'].append(row)
                        output.write_text(json.dumps(report, indent=2) + '\n')
                        print(round_no, profile, suite, round(row['median_ms'], 3), 'ms', flush=True)
                    client.conn.close()
                    if a.check_concurrency:
                        check_path = output.with_name('concurrency-' + profile + '.json')
                        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] /
                            'scripts/amd/test_mi355x_http.py'), '--url', base, '--output', str(check_path)], check=True)
                        report.setdefault('concurrency', {})[profile] = json.loads(check_path.read_text())
                        output.write_text(json.dumps(report, indent=2) + '\n')
                finally:
                    subprocess.run(['docker', 'stop', '--timeout', '5', name], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False)
                    process.wait(timeout=30)


if __name__ == '__main__':
    main()
