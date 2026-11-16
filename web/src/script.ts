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
const EXTRACT_URL = 'http://localhost:8080/api/extract';

const editor = new BlockEditor(requireElement('#block-list'), (doc) => {
    texOutput.textContent = documentToTex(doc);
    schedulePdfRefresh();
});

const PDF_REFRESH_DELAY_MS = 1500;
let pdfRefreshTimer: number | undefined;

function schedulePdfRefresh(): void {
    if (pdfView.hidden || texPane.hidden) {
        return;
    }

    window.clearTimeout(pdfRefreshTimer);
    pdfRefreshTimer = window.setTimeout(refreshPdf, PDF_REFRESH_DELAY_MS);
}

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

// ---------- PDF・画像の読み込み ----------
const MODELS_URL = 'http://localhost:8080/api/models';
const EXTRACT_MODEL_KEY = 'retex.extractModel';
const modelSelect = requireElement<HTMLSelectElement>('#extract-model');
const modelNote = requireElement('#extract-model-note');

/** 読み込みに使えるモデル（ai/evaluation/models.json）。評価サイトで選べるものと同じ */
interface ExtractModel {
    id: string;
    name: string;
    input: 'pdf' | 'image';   // image のモデルは画像も読める（PDFはページごとに画像にして読む）
    outputs: string[];
    note?: string;
}

let extractModels: ExtractModel[] = [];

/** 選んでいるモデル。一覧を取れていなければ undefined（サーバーの既定のモデルで読む） */
function selectedModel(): ExtractModel | undefined {
    return extractModels.find((m) => m.id === modelSelect.value);
}

function showModelNote(): void {
    const model = selectedModel();
    if (!model) return;
    const reads = model.input === 'image' ? 'PDF・画像を読めます' : 'PDFだけ読めます';
    modelNote.textContent = [reads, model.note].filter(Boolean).join(' ／ ');
    try {
        localStorage.setItem(EXTRACT_MODEL_KEY, model.id);
    } catch {
        // 保存できなくても動作には影響しない
    }
}

async function loadModels(): Promise<void> {
    try {
        const res = await fetch(MODELS_URL);
        if (!res.ok) throw new Error(await res.text());
        extractModels = await res.json();
    } catch (err) {
        console.error(err);
        modelSelect.replaceChildren(new Option('サーバーに接続できません'));
        modelNote.textContent = 'server/ で npm run dev を起動してから、ページを開き直してください。';
        return;
    }
    modelSelect.replaceChildren(...extractModels.map((m) => new Option(m.name, m.id)));
    let saved: string | null = null;
    try {
        saved = localStorage.getItem(EXTRACT_MODEL_KEY);
    } catch {
        // 読めなければ先頭のモデルにする
    }
    if (saved && extractModels.some((m) => m.id === saved)) modelSelect.value = saved;
    modelSelect.disabled = false;
    showModelNote();
}

modelSelect.addEventListener('change', showModelNote);
loadModels();

/** PDF・画像をコンパイルサーバーの /api/extract に送り、選んだモデルでエディタ用の文書にしてもらう。失敗したら理由を投げる */
async function extractFile(file: File, model: ExtractModel | undefined): Promise<BlockDocument> {
    const url = model ? `${EXTRACT_URL}?model=${encodeURIComponent(model.id)}` : EXTRACT_URL;
    let res: Response;
    try {
        // file をそのまま送ると Content-Type: application/pdf などが付き、プリフライト（下見）で止められるため、バイト列で送る
        res = await fetch(url, { method: 'POST', body: await file.arrayBuffer() });
    } catch {
        throw new Error('サーバーに接続できませんでした。server/ で npm run dev を起動しているか確認してください。');
    }
    if (!res.ok) {
        throw new Error(await res.text());
    }

    // サーバーは {"labeled": ラベル付きIR, "document": エディタ用JSON} を返す
    const doc: BlockDocument = (await res.json()).document;
    if (!isCurrentSchema(doc)) {
        throw new Error(`サーバーが返した文書の形式（schemaVersion ${doc.schemaVersion}）がエディタと合いません。ai/common/blocks.py を確認してください。`);
    }
    // サーバーは一時ファイル名（input.pdf など）で読むので、文書名は選んだファイル名にする
    doc.title = file.name;
    return doc;
}

fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    fileInput.value = '';

    const model = selectedModel();
    // PDFしか読めないモデルで画像を選んだときは、送る前に知らせる（サーバーも同じ確認をする）
    if (file.type !== 'application/pdf' && model?.input === 'pdf') {
        alert(`「${model.name}」はPDFしか読めません。\n画像を読むときは、「PDF・画像を読めます」と出るモデルを選んでください。`);
        return;
    }

    fileName.textContent = `${file.name} を読み込み中…`;
    modelSelect.disabled = true;
    try {
        showEditor(await extractFile(file, model));
        fileName.textContent = file.name;
    } catch (err) {
        console.error(err);
        fileName.textContent = 'ファイルを選択して編集を始める';
        alert(`読み込めませんでした。\n${err instanceof Error ? err.message : String(err)}`);
    } finally {
        modelSelect.disabled = extractModels.length === 0;
    }
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

    if (isPdf) {
        refreshPdf();
    }
}

tabTex.addEventListener('click', () => showTab('tex'));
tabPdf.addEventListener('click', () => showTab('pdf'));

// 表示中のPDFの住所。次の更新で古いものをメモリから解放するために覚えておく
let pdfUrl: string | undefined;
let pdfAbort: AbortController | undefined;

/** ファイルの中身を base64 の文字列にする（JSON に入れて送るため） */
function readAsBase64(file: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        // 結果は 'data:image/png;base64,....' の形なので、カンマより後ろだけを取り出す
        reader.onload = () => resolve(String(reader.result).split(',')[1] ?? '');
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
    });
}

/** サーバーに届かなかった失敗。TeX の書き間違いなど、サーバーが理由を返した失敗と見分けるため */
class ConnectionError extends Error {}

/**
 * 今の文書をサーバーでPDFにする。プレビューとダウンロードの両方で使う。
 * 失敗したら、画面に出す理由を message に入れて投げる。
 */
async function compilePdf(signal?: AbortSignal): Promise<Blob> {
    const tex = texOutput.textContent ?? '';
    // \includegraphics で使う画像ファイルも、TeX と一緒に送る
    const { files, missing } = editor.imagesForPreview();
    if (missing.length > 0) {
        throw new Error(`画像ファイルが手元にありません: ${missing.join(', ')}\n`
            + '保存した文書を開き直したあとは、画像ブロックでファイルを選び直してください。');
    }
    const images = await Promise.all(
        files.map(async ({ name, file }) => ({ name, data: await readAsBase64(file) })),
    );
    // 画像を読んでいる間に止められたら、送らずにやめる
    signal?.throwIfAborted();

    let res: Response;
    try {
        res = await fetch(COMPILE_URL, {
            method: 'POST',
            body: JSON.stringify({ tex: tex, images: images }),
            signal: signal,
        });
    } catch (err) {
        if (signal?.aborted) throw err;
        throw new ConnectionError('サーバーに接続できません。server/ で npm run dev を起動しているか確認してください。');
    }

    if (!res.ok) {
        throw new Error(await res.text());
    }
    return await res.blob();
}

async function refreshPdf(): Promise<void> {
    if (pdfAbort) {
        pdfAbort.abort();
    }

    const controller = new AbortController();
    pdfAbort = controller;

    pdfStatus.textContent = '更新中…';
    pdfError.hidden = true;

    try {
        const blob = await compilePdf(controller.signal);

        if (pdfUrl) {        // 前の PDF があったら
            URL.revokeObjectURL(pdfUrl);
        }
        pdfUrl = URL.createObjectURL(blob);
        pdfFrame.src = pdfUrl;

        pdfStatus.textContent = '';
    } catch (err) {
        if (controller.signal.aborted) {
            return;     // 次の更新が始まったので、この結果は捨てる
        }
        console.error(err);
        if (err instanceof ConnectionError) {
            pdfStatus.textContent = 'サーバーに接続できません';
            return;
        }
        pdfError.textContent = err instanceof Error ? err.message : String(err);
        pdfError.hidden = false;
        pdfStatus.textContent = 'エラー';
    }
}

refreshPdfButton.addEventListener('click', refreshPdf);

// ---------- PDFのダウンロード ----------
const downloadPdfButton = requireElement<HTMLButtonElement>('#download-pdf');

/** 文書名からダウンロードするファイル名を作る。拡張子を外し、ファイル名に使えない文字は _ にする */
function pdfFileName(title: string): string {
    const base = title.trim().replace(/\.[^.]*$/, '').replace(/[\\/:*?"<>|]/g, '_');
    return `${base || 'document'}.pdf`;
}

async function downloadPdf(): Promise<void> {
    const label = downloadPdfButton.textContent;
    downloadPdfButton.disabled = true;
    downloadPdfButton.textContent = '作成中…';
    try {
        const blob = await compilePdf();
        const url = URL.createObjectURL(blob);
        // download 属性付きのリンクを押したことにすると、ブラウザが保存してくれる
        const link = document.createElement('a');
        link.href = url;
        link.download = pdfFileName(editor.title);
        link.click();
        // すぐ消すと保存が始まる前に消えるブラウザがあるので、少し待つ
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
        console.error(err);
        alert(`PDFを作れませんでした。\n${err instanceof Error ? err.message : String(err)}`);
    } finally {
        downloadPdfButton.disabled = false;
        downloadPdfButton.textContent = label;
    }
}

downloadPdfButton.addEventListener('click', downloadPdf);

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
