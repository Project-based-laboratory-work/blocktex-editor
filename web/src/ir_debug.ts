const irFileInput = document.querySelector<HTMLInputElement>('#json-file')!;
const irOutput = document.querySelector<HTMLDivElement>('#ir-json')!;
const EXTRACT_URL = 'http://localhost:8080/api/extract';
const irPdfInput = document.querySelector<HTMLInputElement>('#pdf-file')!;

function renderIr(ir: any) {
    irOutput.replaceChildren();

    for (const page of ir.pages) {
        for (const el of page.elements) {
            const p = document.createElement('p');
            p.textContent = el.text ?? `[${el.kind}]`;
            irOutput.append(p);
        }
    }
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

    const res = await fetch(EXTRACT_URL, {
        method: 'POST',
        body: await file.arrayBuffer(),
    });

    if (!res.ok) {
        irOutput.textContent = `エラー: ${await res.text()}`;
        return;
    }

    const ir = await res.json();
    renderIr(ir);
});