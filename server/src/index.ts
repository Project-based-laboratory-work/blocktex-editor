import http from 'node:http';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

// execFile を await で待てるようにする
const execFileAsync = promisify(execFile);

const server = http.createServer((req, res) => {
    if (req.method === 'POST' && req.url === '/api/compile') {
        let body = '';
        req.setEncoding('utf8');

        req.on('data', (chunk) => {
            body += chunk;
        });

        req.on('end', async () => {
            try {
                const data = JSON.parse(body);

                if (typeof data.tex !== 'string') {
                    res.statusCode = 400;
                    res.end('tex must be a string');
                    return;
                }

                const dir = await mkdtemp(path.join(os.tmpdir(), 'retex-'));

                try {
                    await writeFile(path.join(dir, 'main.tex'), data.tex);

                    const texArgs = ['-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'main.tex'];

                    await execFileAsync('uplatex', texArgs, { cwd: dir });
                    await execFileAsync('uplatex', texArgs, { cwd: dir });
                    await execFileAsync('dvipdfmx', ['main.dvi'], { cwd: dir });

                    const pdf = await readFile(path.join(dir, 'main.pdf'));
                    res.setHeader('Content-Type', 'application/pdf');
                    res.end(pdf);
                } finally {
                    await rm(dir, { recursive: true, force: true });
                }
            } catch (err) {
                console.error(err);
                res.statusCode = 400;
                res.end('bad request');
            }
        });

        return;
    }
    res.statusCode = 404;
    res.end('not found');
});

server.listen(8080, () => console.log('listening on :8080'));