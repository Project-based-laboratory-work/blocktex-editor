function requireElement<T extends HTMLElement>(selector: string): T {
    const el = document.querySelector<T>(selector);
    if (!el) throw new Error(`必要な要素が見つかりません: ${selector}`);
    return el;
}

const homeView = requireElement('#home-view');
const editorView = requireElement('#editor-view');
const fileInput = requireElement<HTMLInputElement>('#file-upload');
const fileName = requireElement('#file-name');
const texPane = requireElement('#tex-pane');
const texOutput = requireElement('#tex-output');
const toggleTexButton = requireElement('#toggle-tex');
const palette = requireElement('#palette');
const togglePaletteButton = requireElement('#toggle-palette');

const TEX_PANE_KEY = 'retex.texPaneOpen';
const PALETTE_KEY = 'retex.paletteOpen';

const editor = new BlockEditor(requireElement('#block-list'), (doc) => {
    texOutput.textContent = documentToTex(doc);
});

function showEditor(doc: BlockDocument): void {
    homeView.hidden = true;
    editorView.hidden = false;
    editor.load(doc);
}

function showHome(): void {
    editorView.hidden = true;
    homeView.hidden = false;
}

function setPane(pane: HTMLElement, button: HTMLElement, key: string, open: boolean): void {
    pane.hidden = !open;
    button.setAttribute('aria-pressed', String(open));
    try {
        localStorage.setItem(key, String(open));
    } catch {
        // 保存できなくても動作には影響しない
    }
}

const setTexPane = (open: boolean) => setPane(texPane, toggleTexButton, TEX_PANE_KEY, open);
const setPalette = (open: boolean) => setPane(palette, togglePaletteButton, PALETTE_KEY, open);

requireElement('#new-doc').addEventListener('click', () => showEditor(createEmptyDocument()));
requireElement('#back-home').addEventListener('click', showHome);

fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    fileName.textContent = file.name;
    // PDF・画像からのブロック変換は未実装。空のドキュメントで編集を始める。
    showEditor(createEmptyDocument(file.name));
    alert('PDF・画像の自動変換は準備中です。空のドキュメントから編集を始めます。');
    fileInput.value = '';
});

for (const button of document.querySelectorAll<HTMLElement>('[data-add]')) {
    button.addEventListener('click', () => editor.add(button.dataset.add as AddKind));
}

const paletteHint = requireElement('#palette-hint');
let paletteHintTimer: number | undefined;

function insertMathIntoText(): void {
    if (insertInlineMath()) return;
    paletteHint.hidden = false;
    window.clearTimeout(paletteHintTimer);
    paletteHintTimer = window.setTimeout(() => (paletteHint.hidden = true), 3000);
}

const insertMathButton = requireElement('#insert-inline-math');
// mousedown を止めて、段落内のカーソル位置（フォーカス）を保ったままボタンを押せるようにする
insertMathButton.addEventListener('mousedown', (e) => e.preventDefault());
insertMathButton.addEventListener('click', insertMathIntoText);

toggleTexButton.addEventListener('click', () => setTexPane(texPane.hidden));
togglePaletteButton.addEventListener('click', () => setPalette(palette.hidden));

requireElement('#copy-tex').addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(texOutput.textContent ?? '');
    } catch {
        alert('コピーできませんでした。TeXを手動で選択してください。');
    }
});

document.addEventListener('keydown', (e) => {
    if (editorView.hidden || !(e.ctrlKey || e.metaKey)) return;
    if (e.key === '/') {
        e.preventDefault();
        setTexPane(texPane.hidden);
    } else if (e.key === 'm') {
        e.preventDefault();
        insertMathIntoText();
    } else if (e.key === 'b') {
        e.preventDefault();
        setPalette(palette.hidden);
    }
});

// 狭い画面ではパレットを初期は閉じる(オーバーレイ表示のため)
const narrow = window.matchMedia('(max-width: 720px)').matches;
try {
    setTexPane(localStorage.getItem(TEX_PANE_KEY) === 'true');
    setPalette(!narrow && localStorage.getItem(PALETTE_KEY) !== 'false');
} catch {
    // localStorage が使えない環境ではデフォルト(TeX閉・パレット開)
    setTexPane(false);
    setPalette(!narrow);
}
