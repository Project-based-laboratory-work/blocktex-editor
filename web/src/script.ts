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
const tabTex = requireElement('#tab-tex');
const tabPdf = requireElement('#tab-pdf');
const pdfView = requireElement('#pdf-view');
const refreshPdfButton = requireElement('#refresh-pdf');
const pdfStatus = requireElement('#pdf-status');
const pdfError = requireElement('#pdf-error');
const pdfFrame = requireElement<HTMLIFrameElement>('#pdf-frame');

const TEX_PANE_KEY = 'retex.texPaneOpen';
const PALETTE_KEY = 'retex.paletteOpen';
const COMPILE_URL = 'http://localhost:8080/api/compile';

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
const formatHint = requireElement('#format-hint');
const hintTimers = new Map<HTMLElement, number>();

/** 文章の外でボタンを押したときに、数秒だけ案内を出す */
function showHint(hint: HTMLElement): void {
    hint.hidden = false;
    window.clearTimeout(hintTimers.get(hint));
    hintTimers.set(hint, window.setTimeout(() => (hint.hidden = true), 3000));
}

function insertMathIntoText(): void {
    if (!insertInlineMath()) showHint(paletteHint);
}

/** パレットのボタン。mousedown を止めて、段落内のカーソル位置（フォーカス）を保ったまま押せるようにする */
function paletteAction(button: HTMLElement, action: () => void): void {
    button.addEventListener('mousedown', (e) => e.preventDefault());
    button.addEventListener('click', action);
}

paletteAction(requireElement('#insert-inline-math'), insertMathIntoText);
paletteAction(requireElement('#insert-footnote'), () => {
    if (!insertFootnote()) showHint(formatHint);
});
paletteAction(requireElement('#insert-link'), () => {
    if (!editInlineLink()) showHint(formatHint);
});

const formatButtons = document.querySelectorAll<HTMLElement>('[data-format]');
for (const button of formatButtons) {
    paletteAction(button, () => {
        if (!applyInlineFormat(button.dataset.format as FormatKind)) showHint(formatHint);
        updateFormatButtons();
    });
}

/** カーソル位置が太字などかどうかを、パレットのボタンの押下表示に反映する */
function updateFormatButtons(): void {
    const state = inlineFormatState();
    for (const button of formatButtons) {
        const pressed = state?.[button.dataset.format as FormatKind] ?? false;
        button.setAttribute('aria-pressed', String(pressed));
    }
}

document.addEventListener('selectionchange', updateFormatButtons);

// ---------- 文書設定 ----------
const settingsDialog = requireElement<HTMLDialogElement>('#settings-dialog');
const settingPaper = requireElement<HTMLSelectElement>('#setting-paper');
const settingOrientation = requireElement<HTMLSelectElement>('#setting-orientation');
const settingFontSize = requireElement<HTMLSelectElement>('#setting-font-size');
const settingColumns = requireElement<HTMLSelectElement>('#setting-columns');
const settingMarginOn = requireElement<HTMLInputElement>('#setting-margin-on');
const settingMargin = requireElement<HTMLInputElement>('#setting-margin');

/** 余白を指定し始めたときの初期値（mm） */
const DEFAULT_MARGIN_MM = 25;

for (const [value, spec] of Object.entries(PAPER_SIZES)) {
    settingPaper.append(new Option(spec.label, value));
}
for (const size of FONT_SIZES) {
    settingFontSize.append(new Option(`${size}pt`, String(size)));
}
settingMargin.min = String(MARGIN_MM_MIN);
settingMargin.max = String(MARGIN_MM_MAX);

/** 開いている文書の設定をフォームに入れる */
function fillSettingsForm(): void {
    const s = { ...DEFAULT_SETTINGS, ...editor.settings };
    settingPaper.value = s.paper;
    settingOrientation.value = s.landscape ? 'landscape' : 'portrait';
    settingFontSize.value = String(s.fontSize);
    settingColumns.value = s.twoColumn ? '2' : '1';
    settingMarginOn.checked = editor.settings.marginMm !== undefined;
    settingMargin.value = String(editor.settings.marginMm ?? DEFAULT_MARGIN_MM);
    settingMargin.disabled = !settingMarginOn.checked;
}

/** 余白の入力値。空や範囲外のときは範囲内に収めた値を返す */
function marginFromForm(): number {
    const value = Math.round(Number(settingMargin.value));
    if (!Number.isFinite(value) || settingMargin.value === '') return DEFAULT_MARGIN_MM;
    return Math.min(MARGIN_MM_MAX, Math.max(MARGIN_MM_MIN, value));
}

// 既定値と同じ値は保存しない（undefined にして消す）。既定値が変わったときに古い文書も追従させるため
settingPaper.addEventListener('change', () => {
    const paper = settingPaper.value as PaperSize;
    editor.updateSettings({ paper: paper === DEFAULT_SETTINGS.paper ? undefined : paper });
});
settingOrientation.addEventListener('change', () => {
    editor.updateSettings({ landscape: settingOrientation.value === 'landscape' || undefined });
});
settingFontSize.addEventListener('change', () => {
    const size = Number(settingFontSize.value) as FontSize;
    editor.updateSettings({ fontSize: size === DEFAULT_SETTINGS.fontSize ? undefined : size });
});
settingColumns.addEventListener('change', () => {
    editor.updateSettings({ twoColumn: settingColumns.value === '2' || undefined });
});
settingMarginOn.addEventListener('change', () => {
    settingMargin.disabled = !settingMarginOn.checked;
    editor.updateSettings({ marginMm: settingMarginOn.checked ? marginFromForm() : undefined });
});
settingMargin.addEventListener('input', () => {
    if (settingMarginOn.checked && settingMargin.value !== '') editor.updateSettings({ marginMm: marginFromForm() });
});
// 入力欄を離れたら、範囲外の値を実際に使う値に書き直して見せる
settingMargin.addEventListener('change', () => {
    settingMargin.value = String(marginFromForm());
});

requireElement('#open-settings').addEventListener('click', () => {
    fillSettingsForm();
    settingsDialog.showModal();
});
// 枠の外（背景）をクリックしても閉じる
settingsDialog.addEventListener('click', (e) => {
    if (e.target === settingsDialog) settingsDialog.close();
});

function showTab(tab: 'tex' | 'pdf'): void {
    const isPdf = tab === 'pdf';      // PDF タブなら true

    tabTex.setAttribute('aria-selected', String(!isPdf));
    tabPdf.setAttribute('aria-selected', String(isPdf));
    texOutput.hidden = isPdf;
    pdfView.hidden = !isPdf;
}

tabTex.addEventListener('click', () => showTab('tex'));
tabPdf.addEventListener('click', () => showTab('pdf'));

async function refreshPdf(): Promise<void> {
    pdfStatus.textContent = '更新中…';
    pdfError.hidden = true;

    try {
        const tex = texOutput.textContent ?? '';
        const res = await fetch(COMPILE_URL, {
            method: 'POST',
            body: JSON.stringify({ tex: tex }),
        });

        if (!res.ok) {
            pdfError.textContent = await res.text();
            pdfError.hidden = false;
            pdfStatus.textContent = 'エラー';
            return;
        }

        const blob = await res.blob();
        pdfFrame.src = URL.createObjectURL(blob);
        pdfStatus.textContent = '';
    } catch (err) {
        console.error(err);
        pdfStatus.textContent = 'サーバーに接続できません';
    }
}

refreshPdfButton.addEventListener('click', refreshPdf);

toggleTexButton.addEventListener('click', () => setTexPane(texPane.hidden));
togglePaletteButton.addEventListener('click', () => setPalette(palette.hidden));

requireElement('#copy-tex').addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(texOutput.textContent ?? '');
    } catch {
        alert('コピーできませんでした。TeXを手動で選択してください。');
    }
});

// Ctrl+B / I / U / K は文章の入力中なら richtext.ts 側で装飾として処理し、ここまで届かない
document.addEventListener('keydown', (e) => {
    if (editorView.hidden || settingsDialog.open || !(e.ctrlKey || e.metaKey)) return;
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
