import http from 'node:http';
import { mkdtemp, writeFile, readFile, rm, access } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

// execFile を await で待てるようにする
const execFileAsync = promisify(execFile);

// パラメータ
const MAX_BODY = 20_000_000;   // 文字数制限(約20MB)。画像を base64 で一緒に送るため大きめ
const COMPILE_TIMEOUT_MS = 20_000; // タイムアウト
const PORT = Number(process.env.PORT ?? 8080);
// server/src/index.ts → ../../ai が ai/ フォルダ
const AI_DIR = path.resolve(import.meta.dirname, '../../ai');
const PYTHON = path.join(AI_DIR, '.venv/bin/python');

// 各自のPCで動かす前提なので、画面を開く場所（file:// で開くと 'null'、または localhost）からだけ受け付ける。
// ほかのサイトを開いているときに、そのページから勝手にコンパイルさせられないようにするため。
const ALLOWED_ORIGIN = /^(null|https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?)$/;

// 画像のファイル名。フォルダの区切り・制御文字・先頭の . を含まず、dvipdfmx で読める拡張子のもの。
// main.* は作業用のファイル（main.tex や出力の main.pdf）とぶつかるので受け付けない。
const IMAGE_NAME = /^(?!main\.)[^/\\.\x00-\x1f][^/\\\x00-\x1f]*\.(png|jpe?g|pdf)$/i;

interface ImageFile {
    name: string;
    data: string;   // base64
}

function isImageFile(x: any): x is ImageFile {
    return typeof x?.name === 'string' && IMAGE_NAME.test(x.name) && typeof x.data === 'string';
}

const server = http.createServer((req, res) => {
    const origin = req.headers.origin;
    if (origin !== undefined) {
        if (!ALLOWED_ORIGIN.test(origin)) {
            res.statusCode = 403;
            res.end('forbidden');
            return;
        }
        res.setHeader('Access-Control-Allow-Origin', origin);
    }
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

                const images = data.images ?? [];
                if (!Array.isArray(images)) {
                    res.statusCode = 400;
                    res.end('images must be an array');
                    return;
                }
                const badImage = images.find((x) => !isImageFile(x));
                if (badImage !== undefined) {
                    res.statusCode = 400;
                    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
                    res.end(`この画像のファイル名は使えません: ${String(badImage?.name)}（png / jpg / pdf で、main. で始まらない名前にしてください）`);
                    return;
                }

                const dir = await mkdtemp(path.join(os.tmpdir(), 'retex-'));

                try {
                    await writeFile(path.join(dir, 'main.tex'), data.tex);

                    // \includegraphics は画像の大きさを extractbb で調べるが、-no-shell-escape だと
                    // TeX からは呼べない。先にこちらで実行して .xbb を作っておく。
                    for (const image of images) {
                        await writeFile(path.join(dir, image.name), Buffer.from(image.data, 'base64'));
                        try {
                            await execFileAsync('extractbb', [image.name], { cwd: dir, timeout: COMPILE_TIMEOUT_MS });
                            // 壊れた画像だと、extractbb は成功扱いで終わって .xbb を作らないことがある
                            await access(path.join(dir, `${path.parse(image.name).name}.xbb`));
                        } catch {
                            res.statusCode = 422;
                            res.setHeader('Content-Type', 'text/plain; charset=utf-8');
                            res.end(`画像 ${image.name} を読み込めませんでした`);
                            return;
                        }
                    }

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
                        // 送られていない画像を \includegraphics すると、graphicx が extractbb を呼ぼうとしてこのエラーになる
                        if (message.includes('Cannot run pipe command')) {
                            message = '画像ファイルが見つかりません。画像ブロックでファイルを選び直してください';
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

    if (req.method === 'POST' && req.url === '/api/extract') {
        const chunks: Buffer[] = [];
        let size = 0;

        req.on('data', (chunk: Buffer) => {
            chunks.push(chunk);
            size += chunk.length;
            if (size > MAX_BODY) {
                res.statusCode = 413;
                res.end('too large');
                req.destroy();
            }
        });

        req.on('end', async () => {
            try {
                const pdf = Buffer.concat(chunks);
                // PDFファイルは必ず "%PDF-" で始まる。違うものは抽出機に渡す前に断る
                if (pdf.subarray(0, 5).toString('latin1') !== '%PDF-') {
                    res.statusCode = 422;
                    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
                    res.end('PDFファイルではありません');
                    return;
                }

                const dir = await mkdtemp(path.join(os.tmpdir(), 'extract-'));

                try {
                    const pdfPath = path.join(dir, 'input.pdf');
                    await writeFile(pdfPath, pdf);

                    let stdout: string;
                    try {
                        ({ stdout } = await execFileAsync(
                            PYTHON,
                            // 抽出 → 分類 → エディタ用JSON への変換までをまとめて行い、
                            // {"labeled": ラベル付きIR, "document": エディタ用JSON} を標準出力に出す
                            ['-m', 'rule_based.ir_adapter', pdfPath, '--stdout'],
                            { cwd: AI_DIR, timeout: COMPILE_TIMEOUT_MS, maxBuffer: MAX_BODY },
                        ));
                    } catch (err: any) {
                        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
                        // ai/.venv を作っていないと、Python 自体が見つからない
                        if (err.code === 'ENOENT') {
                            console.error(err);
                            res.statusCode = 500;
                            res.end('抽出機のPythonが見つかりません（ai/.venv を作ってください）');
                            return;
                        }
                        if (err.killed) {
                            res.statusCode = 422;
                            res.end('抽出に時間がかかりすぎたため中断しました');
                            return;
                        }
                        // 壊れたPDFなど。詳しい経緯（Traceback）はサーバーの端末に出す
                        console.error(err.stderr);
                        res.statusCode = 422;
                        res.end('PDFを読み込めませんでした');
                        return;
                    }

                    res.setHeader('Content-Type', 'application/json; charset=utf-8');
                    res.end(stdout);
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

server.listen(PORT, () => console.log(`listening on :${PORT}`));