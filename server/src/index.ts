import http from 'node:http';

const server = http.createServer((req, res) => {
    if (req.method === 'POST' && req.url === '/api/compile') {
        let body = '';
        req.setEncoding('utf8');

        req.on('data', (chunk) => {
            body += chunk;
        });

        req.on('end', () => {
            try {
                const data = JSON.parse(body);

                if (typeof data.tex !== 'string') {
                    res.statusCode = 400;
                    res.end('tex must be a string');
                    return;
                }

                res.end(String(data.tex.length));
            } catch {
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