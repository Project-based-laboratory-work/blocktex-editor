import http from 'node:http';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

// execFile を await で待てるようにする
const execFileAsync = promisify(execFile);

// パラメータ
const MAX_BODY = 1_000_000;   // 文字数制限(約1MB)
const COMPILE_TIMEOUT_MS = 20_000; // タイムアウト

const server = http.createServer((req, res) => {
    if (req.method === 'POST' && req.url === '/api/compile') {
        let body = '';
        req.setEncoding('utf8');

        req.on('data', (chunk) => {
            body += chunk;
            if (body.length > MAX_BODY) {
                res.statusCode = 413;
                res.end('too large');
                req.destroy();
            }
        });

        req.on('end', async () => {
            try {
                let data;
                try {
                    data = JSON.parse(body);
                } catch {
                    res.statusCode = 400;
                    res.end('bad request');
                    return;
                }

                if (typeof data.tex !== 'string') {
                    res.statusCode = 400;
                    res.end('tex must be a string');
                    return;
                }

                const dir = await mkdtemp(path.join(os.tmpdir(), 'retex-'));

                try {
                    await writeFile(path.join(dir, 'main.tex'), data.tex);

                    const texArgs = ['-interaction=nonstopmode', '-halt-on-error', '-no-shell-escape', 'main.tex'];

                    try {
                        await execFileAsync('uplatex', texArgs, { cwd: dir, timeout: COMPILE_TIMEOUT_MS });
                        await execFileAsync('uplatex', texArgs, { cwd: dir, timeout: COMPILE_TIMEOUT_MS });
                    } catch (err: any) {
                        // タイムアウトした際の処理
                        if (err.killed) {
                            res.statusCode = 422;
                            res.setHeader('Content-Type', 'text/plain; charset=utf-8');
                            res.end('コンパイルに時間がかかりすぎたため中断しました');
                            return;
                        }

                        // TeXの書き間違いは 422 で返し、ユーザーに原因を見せる
                        const log = await readFile(path.join(dir, 'main.log'), 'utf8');
                        const lines = log.split('\n');
                        const start = lines.findIndex((line) => line.startsWith('!'));
                        let message = '';
                        if (start === -1) {
                            message = lines.slice(-20).join('\n');
                        } else {
                            message = lines.slice(start, start + 3).join('\n');
                        }

                        res.statusCode = 422;
                        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
                        res.end(message);
                        return;
                    }

                    await execFileAsync('dvipdfmx', ['main.dvi'], { cwd: dir, timeout: COMPILE_TIMEOUT_MS });

                    const pdf = await readFile(path.join(dir, 'main.pdf'));
                    res.setHeader('Content-Type', 'application/pdf');
                    res.end(pdf);
                } finally {
                    await rm(dir, { recursive: true, force: true });
                }
            } catch (err) {
                console.error(err);
                res.statusCode = 500;
                res.end('server error');
            }
        });

        return;
    }
    res.statusCode = 404;
    res.end('not found');
});

server.listen(8080, () => console.log('listening on :8080'));