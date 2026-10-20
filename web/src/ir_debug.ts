const irFileInput = document.querySelector<HTMLInputElement>('#json-file')!;

irFileInput.addEventListener('change', async () => {
    const file = irFileInput.files?.[0];
    if (file === undefined) {
        return;
    }

    const text = await file.text();
    const ir = JSON.parse(text);
    console.log(ir);
    const irOutput = document.querySelector<HTMLDivElement>('#ir-json')!;

    irOutput.replaceChildren();

    for (const page of ir.pages) {
        for (const el of page.elements) {
            const p = document.createElement('p');
            p.textContent = el.text ?? `[${el.kind}]`;
            irOutput.append(p);
        }
    }

});
