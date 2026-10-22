const irFileInput = document.querySelector<HTMLInputElement>('#json-file')!;
const irOutput = document.querySelector<HTMLDivElement>('#ir-json')!;
const IR_DEBUG_EXTRACT_URL = 'http://localhost:8080/api/extract';
const irPdfInput = document.querySelector<HTMLInputElement>('#pdf-file')!;

function renderIr(ir: any) {
    irOutput.replaceChildren();

    // 1行に1要素。ラベル付きIRなら label（heading など）、無ければ kind（text など）を出す
    for (const page of ir.pages) {
        for (const el of page.elements) {
            const p = document.createElement('p');
            p.textContent = `[${el.label ?? el.kind}] ${el.text ?? ''}`;
            irOutput.append(p);
        }
    }

    // bbox や font_size などの細かい値は、JSON をそのまま出して確認する
    const rawTitle = document.createElement('h2');
    rawTitle.textContent = 'IR（全体）';
    const raw = document.createElement('pre');
    raw.textContent = JSON.stringify(ir, null, 2);
    irOutput.append(rawTitle, raw);
}

irFileInput.addEventListener('change', async () => {
    const file = irFileInput.files?.[0];
    if (file === undefined) {
        return;
    }

    const text = await file.text();
    const ir = JSON.parse(text);
    renderIr(ir);
});

irPdfInput.addEventListener('change', async () => {
    const file = irPdfInput.files?.[0];
    if (file === undefined) {
        return;
    }

    let res: Response;
    try {
        res = await fetch(IR_DEBUG_EXTRACT_URL, {
            method: 'POST',
            body: await file.arrayBuffer(),
        });
    } catch (err) {
        // サーバーが動いていないと、返事が来ずに fetch 自体がエラーになる
        console.error(err);
        irOutput.textContent = 'サーバーに接続できませんでした。server/ で npm run dev を起動しているか確認してください';
        return;
    }

    if (!res.ok) {
        irOutput.textContent = `エラー: ${await res.text()}`;
        return;
    }

    // サーバーは {"labeled": ラベル付きIR, "document": エディタ用JSON} を返す
    const data = await res.json();
    renderIr(data.labeled);

    const documentTitle = document.createElement('h2');
    documentTitle.textContent = 'エディタ用JSON（document）';
    const documentJson = document.createElement('pre');
    documentJson.textContent = JSON.stringify(data.document, null, 2);
    irOutput.append(documentTitle, documentJson);
});
