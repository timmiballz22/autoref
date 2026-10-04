import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import chromium from '@sparticuz/chromium';

const remotion = fileURLToPath(new URL('../node_modules/.bin/remotion', import.meta.url));
const browserExecutable = await chromium.executablePath();

const result = spawnSync(
  remotion,
  [
    'render',
    'src/index.ts',
    'HumanInnovation',
    'out/human-innovation.mp4',
    `--browser-executable=${browserExecutable}`,
  ],
  {stdio: 'inherit'},
);

if (result.error) {
  throw result.error;
}

process.exit(result.status ?? 1);
