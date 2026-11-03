// 評価サイト（eval.html）の画面。データは server/ の /api/eval/... から読む（ai/evaluation/ が作った結果）。
// 画面は URL の # の後ろで切り替える:
//   #/                          結果一覧（と、評価の実行）
//   #/run/<run_id>              モデル詳細（1回分の評価）
//   #/run/<run_id>/case/<id>    ケース詳細（ページに正解と予測の枠を重ねる）
//
// ほかの画面のスクリプトと名前がぶつからないように、全体を関数の中に入れている
// （web/src は import/export を使わないグローバルスクリプト方式のため）。
(() => {
    const API = 'http://localhost:8080/api/eval';

    type Scores = Record<string, number | null>;

    interface RunSummary {
        run_id: string;
        model: { id: string; name: string };
        dataset: string;
        summary: Scores;
        errors: number;
    }

    interface ModelInfo {
        id: string;
        name: string;
        input: string;
        outputs: string[];
        note?: string;
    }

    interface DatasetInfo {
        name: string;
        created: string;
        cases: number;
        tags: Record<string, number>;
    }

    interface Finding {
        kind: 'strength' | 'weakness';
        metric: string;
        slice: string;
        value: number;
        overall: number;
        n: number;
        text: string;
    }

    interface CaseReport {
        id: string;
        tags: Record<string, string | number | boolean>;
        seconds: number;
        pages: number;
        scores: Scores;
        error: string | null;
        pred_path: string;
    }

    interface Report {
        run_id: string;
        created?: string;
        model: { id: string; name: string; input?: string; outputs: string[]; params: Record<string, unknown> };
        dataset: string;
        iou_threshold: number;
        summary: Scores;
        by_tag: Record<string, Scores>;
        tag_cases?: Record<string, number>;
        findings?: Finding[];
        confusion: { labels: string[]; matrix: number[][] } | null;
        errors?: number;
        cases: CaseReport[];
    }

    // common/ir.py のラベル付きIR（画面で使う項目だけ）
    interface IrRun {
        text: string;
        bold?: boolean;
        italic?: boolean;
        underline?: boolean;
    }

    interface IrElement {
        kind: string;
        bbox: [number, number, number, number];
        text?: string;
        runs?: IrRun[];
        label?: string;
        level?: number;
    }

    interface IrPage {
        page: number;
        width: number;
        height: number;
        unit: string;
        elements: IrElement[];
    }

    interface IrDocument {
        pages: IrPage[];
    }

    // 画面に出す指標。better: 大きいほど良い（'high'）か、小さいほど良い（'low'）か
    interface MetricInfo {
        key: string;
        name: string;
        better: 'high' | 'low';
        help: string;
    }

    const METRICS: MetricInfo[] = [
        { key: 'f1', name: 'ブロック検出 F1', better: 'high', help: '見つけたブロックの正しさ（Precision）と、見逃しの少なさ（Recall）の調和平均' },
        { key: 'precision', name: 'Precision', better: 'high', help: '出したブロックのうち、正解と重なった割合（誤検出が多いと下がる）' },
        { key: 'recall', name: 'Recall', better: 'high', help: '正解のブロックのうち、見つけられた割合（見逃しが多いと下がる）' },
        { key: 'miou', name: '枠の位置 mIoU', better: 'high', help: '当たったブロックの、正解との重なり具合（IoU）の平均' },
        { key: 'label_acc', name: 'ラベル正解率', better: 'high', help: '当たったブロックのうち、ラベル（見出し・段落…）が合っていた割合' },
        { key: 'level_acc', name: '見出しレベル正解率', better: 'high', help: '見出し同士の組のうち、見出しレベル（1〜3）が合っていた割合' },
        { key: 'cer', name: '文字誤り率 CER', better: 'low', help: '直す必要のある文字の数 ÷ 正解の文字数（空白・改行は数えない）' },
        { key: 'cer_ja', name: 'CER（日本語）', better: 'low', help: '日本語などASCII以外の文字だけで数えたCER（並び順の違いは数えない）' },
        { key: 'cer_en', name: 'CER（英語）', better: 'low', help: 'ASCIIの文字だけで数えたCER（並び順の違いは数えない）' },
        { key: 'deco_f1', name: '装飾 F1', better: 'high', help: '段落・箇条書きの太字・斜体・下線が、正しい文字に付いていたか' },
        { key: 'sec_per_page', name: '秒/ページ', better: 'low', help: '1ページを読むのにかかった時間' },
    ];
    const METRIC_BY_KEY = new Map(METRICS.map((m) => [m.key, m]));
    const LIST_METRICS = ['f1', 'miou', 'label_acc', 'cer', 'deco_f1', 'sec_per_page'];

    // 「能力の内訳」の横棒。どれも 0〜1 で、長いほど良いように直して並べる
    const ABILITY_AXES: { name: string; key: string; invert: boolean }[] = [
        { name: 'ブロックの検出', key: 'f1', invert: false },
        { name: '枠の位置', key: 'miou', invert: false },
        { name: 'ラベル', key: 'label_acc', invert: false },
        { name: '見出しレベル', key: 'level_acc', invert: false },
        { name: '文字（全体）', key: 'cer', invert: true },
        { name: '文字（日本語）', key: 'cer_ja', invert: true },
        { name: '文字（英語）', key: 'cer_en', invert: true },
        { name: '装飾', key: 'deco_f1', invert: false },
    ];

    // 切り口（run.py の tag_keys の名前）を読める言葉にする（run.py の TAG_NAMES と同じ）
    const TAG_NAMES: Record<string, string> = {
        'lang=ja': '日本語',
        'lang=en': '英語',
        'has_list=True': '箇条書きあり',
        'has_list=False': '箇条書きなし',
        'has_title=True': 'タイトルあり',
        'has_title=False': 'タイトルなし',
        'has_decoration=True': '装飾あり',
        'has_decoration=False': '装飾なし',
    };

    // 混同行列の濃さ（青の1色で、薄い → 濃い）
    const RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95'];

    const app = document.querySelector<HTMLElement>('#app')!;
    const tooltip = document.querySelector<HTMLDivElement>('#tooltip')!;

    // ---------- 小さな道具 ----------

    type Child = Node | string | null | undefined | false;

    /** 要素を作る。props は属性（class, href など）。文字は textContent として入れるので、HTMLとして解釈されない */
    function h<K extends keyof HTMLElementTagNameMap>(
        tag: K,
        props: Record<string, string | undefined> = {},
        ...children: Child[]
    ): HTMLElementTagNameMap[K] {
        const e = document.createElement(tag);
        for (const [k, v] of Object.entries(props)) {
            if (v !== undefined) {
                e.setAttribute(k, v);
            }
        }
        for (const c of children) {
            if (c !== null && c !== undefined && c !== false) {
                e.append(c);
            }
        }
        return e;
    }

    async function getJson<T>(path: string): Promise<T> {
        let res: Response;
        try {
            res = await fetch(API + path);
        } catch {
            throw new Error('サーバーに接続できませんでした。server/ で npm run dev を起動しているか確認してください');
        }
        if (!res.ok) {
            throw new Error(`${res.status} ${await res.text()}（${path}）`);
        }
        return res.json();
    }

    function format(key: string, value: number | null | undefined): string {
        if (value === null || value === undefined) {
            return '—';
        }
        if (key === 'sec_per_page') {
            return `${value.toFixed(2)}秒`;
        }
        return value.toFixed(3);
    }

    function tagName(key: string): string {
        return TAG_NAMES[key] ?? key.replace(/^pages=(\d+)$/, '$1ページ');
    }

    function runDate(runId: string): string {
        const m = runId.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/);
        return m ? `${m[1]}/${m[2]}/${m[3]} ${m[4]}:${m[5]}` : runId;
    }

    function showTooltip(target: HTMLElement, text: string): void {
        target.addEventListener('mousemove', (e) => {
            tooltip.textContent = text;
            tooltip.hidden = false;
            tooltip.style.left = `${Math.min(e.clientX + 12, window.innerWidth - 330)}px`;
            tooltip.style.top = `${e.clientY + 14}px`;
        });
        target.addEventListener('mouseleave', () => {
            tooltip.hidden = true;
        });
    }

    function errorBox(err: unknown): HTMLElement {
        return h('div', { class: 'error-box' }, err instanceof Error ? err.message : String(err));
    }

    /** 同じデータセットで、モデルごとに一番新しい実行だけを残す（runs は新しい順） */
    function latestPerModel(runs: RunSummary[], dataset: string): RunSummary[] {
        const seen = new Set<string>();
        const result: RunSummary[] = [];
        for (const run of runs) {
            if (run.dataset === dataset && !seen.has(run.model.id)) {
                seen.add(run.model.id);
                result.push(run);
            }
        }
        return result;
    }

    /** runs のうち key の値が一番良いもの。値の無い実行は比べない */
    function bestValue(runs: RunSummary[], key: string): number | null {
        const info = METRIC_BY_KEY.get(key);
        const values = runs.map((r) => r.summary[key]).filter((v): v is number => v !== null && v !== undefined);
        if (values.length === 0 || info === undefined) {
            return null;
        }
        return info.better === 'high' ? Math.max(...values) : Math.min(...values);
    }

    /** value が runs の中で何位か（同じ値は同じ順位）。比べられなければ null */
    function rankOf(runs: RunSummary[], key: string, value: number | null): { rank: number; of: number } | null {
        const info = METRIC_BY_KEY.get(key);
        if (value === null || info === undefined) {
            return null;
        }
        const values = runs.map((r) => r.summary[key]).filter((v): v is number => v !== null && v !== undefined);
        const better = values.filter((v) => (info.better === 'high' ? v > value : v < value)).length;
        return { rank: better + 1, of: values.length };
    }

    // ---------- 結果一覧 ----------

    async function renderHome(): Promise<void> {
        const [runs, models, datasets] = await Promise.all([
            getJson<RunSummary[]>('/runs'),
            getJson<ModelInfo[]>('/models'),
            getJson<DatasetInfo[]>('/datasets'),
        ]);
        app.replaceChildren(runForm(models, datasets), ...resultTables(runs, datasets));
    }

    function runForm(models: ModelInfo[], datasets: DatasetInfo[]): HTMLElement {
        const modelSelect = h('select', { id: 'model' }, ...models.map((m) => h('option', { value: m.id }, m.name)));
        const datasetSelect = h(
            'select',
            { id: 'dataset' },
            ...datasets.map((d) => h('option', { value: d.name }, `${d.name}（${d.cases}件）`)),
        );
        const button = h('button', { class: 'primary', type: 'button' }, '評価を実行');
        const note = h('p', { class: 'note' });
        const status = h('div');

        const updateNote = () => {
            const model = models.find((m) => m.id === modelSelect.value);
            const outputs = model ? `出すもの: ${model.outputs.join(', ')}（入力: ${model.input === 'image' ? 'ページの画像' : 'PDF'}）` : '';
            note.textContent = [outputs, model?.note].filter(Boolean).join(' ／ ');
        };
        modelSelect.addEventListener('change', updateNote);
        updateNote();

        button.addEventListener('click', async () => {
            button.disabled = true;
            status.replaceChildren(h('p', { class: 'meta' }, '実行中…（ルールベースは数秒、OCRのモデルは数分かかります）'));
            try {
                const res = await fetch(`${API}/run`, {
                    method: 'POST',
                    body: JSON.stringify({ model: modelSelect.value, dataset: datasetSelect.value }),
                });
                if (!res.ok) {
                    throw new Error(await res.text());
                }
                const { run_id } = await res.json();
                location.hash = `#/run/${run_id}`;
            } catch (err) {
                status.replaceChildren(errorBox(err));
            } finally {
                button.disabled = false;
            }
        });

        const form = h(
            'div',
            { class: 'run-form' },
            h('label', {}, 'モデル', modelSelect),
            h('label', {}, 'データセット', datasetSelect),
            button,
        );
        const empty = datasets.length === 0
            ? h('p', { class: 'note' }, 'データセットがありません。ai/ で python -m evaluation.make_dataset --out evaluation/data/synth_v1 --docs 20 を実行してください')
            : null;
        return h('section', { class: 'card' }, h('h2', {}, '評価を実行'), h('p', { class: 'lead' }, '登録したモデルを、正解付きのデータセットで採点します。'), form, note, empty, status);
    }

    function resultTables(runs: RunSummary[], datasets: DatasetInfo[]): HTMLElement[] {
        if (runs.length === 0) {
            return [h('section', { class: 'card' }, h('h2', {}, '結果一覧'), h('p', { class: 'note' }, 'まだ評価していません。上のフォームから実行してください。'))];
        }
        const names = [...new Set(runs.map((r) => r.dataset))];
        return names.map((dataset) => {
            const rows = runs.filter((r) => r.dataset === dataset);
            const latest = latestPerModel(runs, dataset);
            const info = datasets.find((d) => d.name === dataset);
            const head = h(
                'tr',
                {},
                h('th', {}, 'モデル'),
                h('th', {}, '実行日時'),
                ...LIST_METRICS.map((key) => {
                    const th = h('th', { class: 'num' }, METRIC_BY_KEY.get(key)!.name);
                    showTooltip(th, METRIC_BY_KEY.get(key)!.help);
                    return th;
                }),
                h('th', { class: 'num' }, '失敗'),
            );
            const body = rows.map((run) => {
                const isLatest = latest.includes(run);
                const cells = LIST_METRICS.map((key) => {
                    const value = run.summary[key] ?? null;
                    const best = isLatest && latest.length > 1 && value !== null && value === bestValue(latest, key);
                    return h('td', { class: `num${value === null ? ' na' : ''}${best ? ' best' : ''}` }, `${best ? '★ ' : ''}${format(key, value)}`);
                });
                const tr = h(
                    'tr',
                    { class: 'clickable' },
                    h('td', {}, run.model.name),
                    h('td', { class: 'meta' }, runDate(run.run_id)),
                    ...cells,
                    h('td', { class: `num${run.errors > 0 ? ' down' : ''}` }, String(run.errors ?? 0)),
                );
                tr.addEventListener('click', () => {
                    location.hash = `#/run/${run.run_id}`;
                });
                return tr;
            });
            return h(
                'section',
                { class: 'card' },
                h('h2', {}, `データセット: ${dataset}`),
                h(
                    'p',
                    { class: 'lead' },
                    `${info ? `${info.cases}件。` : ''}★ はモデルごとの最新の実行の中で一番良い値。— は対象外（そのモデルが出さないもの）。行をクリックすると詳細を開きます。`,
                ),
                h('table', {}, h('thead', {}, head), h('tbody', {}, ...body)),
            );
        });
    }

    // ---------- モデル詳細 ----------

    async function renderRun(runId: string): Promise<void> {
        const [report, runs] = await Promise.all([
            getJson<Report>(`/runs/${runId}/report.json`),
            getJson<RunSummary[]>('/runs'),
        ]);
        const others = latestPerModel(runs, report.dataset);
        if (!others.some((r) => r.run_id === report.run_id)) {
            // 古い実行を開いたときも、自分を比べる相手に入れる
            others.push({ run_id: report.run_id, model: report.model, dataset: report.dataset, summary: report.summary, errors: report.errors ?? 0 });
        }

        const outputs = report.model.outputs.join(', ');
        const header = h(
            'section',
            { class: 'card' },
            h('div', { class: 'breadcrumb' }, h('a', { href: '#/' }, '← 結果一覧')),
            h('h2', {}, `${report.model.name} × ${report.dataset}`),
            h('p', { class: 'meta' }, `実行: ${runDate(report.run_id)} ／ 出すもの: ${outputs} ／ IoUの基準: ${report.iou_threshold} ／ ${report.cases.length}件`),
            (report.errors ?? 0) > 0
                ? h('div', { class: 'error-box' }, `${report.errors}件でモデルが失敗しました（そのケースは「何も見つけられなかった」として採点しています）。下のケース一覧で理由を確認できます。`)
                : null,
        );

        app.replaceChildren(
            header,
            statsCard(report),
            abilityCard(report, others),
            findingsCard(report),
            tagCard(report),
            confusionCard(report),
            casesCard(report),
        );
    }

    function statsCard(report: Report): HTMLElement {
        const keys = ['f1', 'precision', 'recall', 'miou', 'label_acc', 'cer', 'deco_f1', 'sec_per_page'];
        const tiles = keys.map((key) => {
            const info = METRIC_BY_KEY.get(key)!;
            const tile = h('div', { class: 'stat' }, h('div', { class: 'label' }, info.name), h('div', { class: `value${report.summary[key] == null ? ' na' : ''}` }, format(key, report.summary[key])));
            showTooltip(tile, info.help);
            return tile;
        });
        return h('section', { class: 'card' }, h('h2', {}, '全体の成績'), h('p', { class: 'lead' }, '— は対象外（そのモデルが出さないもの、または比べるものが無かったもの）。'), h('div', { class: 'stats' }, ...tiles));
    }

    function abilityCard(report: Report, others: RunSummary[]): HTMLElement {
        const rows: Node[] = [];
        for (const axis of ABILITY_AXES) {
            const raw = report.summary[axis.key] ?? null;
            const score = raw === null ? null : Math.max(0, Math.min(1, axis.invert ? 1 - raw : raw));
            const track = h('div', { class: 'bar-track' });
            if (score !== null) {
                track.append(h('div', { class: 'bar-fill', style: `width: ${(score * 100).toFixed(1)}%` }));
            }
            const hit = h('div', { class: 'bar-hit' });
            track.append(hit);
            const info = METRIC_BY_KEY.get(axis.key)!;
            showTooltip(hit, score === null ? `${axis.name}: 対象外` : `${axis.name}: ${score.toFixed(3)}（${info.name} = ${format(axis.key, raw)}）`);

            const rank = rankOf(others, axis.key, raw);
            rows.push(
                h('div', {}, axis.name),
                track,
                h('div', { class: `bar-value${score === null ? ' na' : ''}` }, score === null ? '対象外' : score.toFixed(3)),
                h('div', { class: 'bar-rank' }, rank && rank.of > 1 ? `${rank.rank}位 / ${rank.of}モデル` : ''),
            );
        }
        return h(
            'section',
            { class: 'card' },
            h('h2', {}, '能力の内訳'),
            h('p', { class: 'lead' }, '0〜1 の長さで、長いほど良い。文字は「1 − 誤り率」に直しています。右の順位は、同じデータセットで評価した各モデルの最新の実行の中での順位です。'),
            h('div', { class: 'bars' }, ...rows),
        );
    }

    function findingsCard(report: Report): HTMLElement {
        const findings = report.findings ?? [];
        const list = findings.length > 0
            ? h(
                'ul',
                { class: 'findings' },
                ...findings.map((f) =>
                    h(
                        'li',
                        {},
                        h('span', { class: `badge ${f.kind === 'strength' ? 'up' : 'down'}` }, f.kind === 'strength' ? '✓ 強み' : '! 弱み'),
                        f.text,
                    ),
                ),
            )
            : h('p', { class: 'note' }, report.findings === undefined
                ? 'この結果は、強み・弱みを出す前の版の評価で作られています。もう一度評価すると表示されます。'
                : '全体と比べて目立つ差（0.05以上、5件以上の切り口）はありませんでした。');
        return h('section', { class: 'card' }, h('h2', {}, '強み・弱み'), h('p', { class: 'lead' }, '文書の種類（切り口）ごとの成績を全体と比べて、差が大きい順に並べています。'), list);
    }

    function tagCard(report: Report): HTMLElement {
        const keys = ['f1', 'miou', 'label_acc', 'cer', 'deco_f1'];
        const head = h('tr', {}, h('th', {}, '切り口'), h('th', { class: 'num' }, '件数'), ...keys.map((k) => h('th', { class: 'num' }, METRIC_BY_KEY.get(k)!.name)));
        const overall = h('tr', {}, h('td', {}, h('strong', {}, '全体')), h('td', { class: 'num' }, String(report.cases.length)), ...keys.map((k) => h('td', { class: 'num' }, format(k, report.summary[k]))));
        const body = Object.keys(report.by_tag).sort().map((tag) => {
            const scores = report.by_tag[tag];
            const cells = keys.map((key) => {
                const value = scores[key] ?? null;
                const base = report.summary[key] ?? null;
                if (value === null || base === null) {
                    return h('td', { class: 'num na' }, format(key, value));
                }
                const info = METRIC_BY_KEY.get(key)!;
                const gain = info.better === 'high' ? value - base : base - value;
                const mark = Math.abs(gain) < 0.05 ? '' : gain > 0 ? ' ▲' : ' ▼';
                return h('td', { class: `num${mark === ' ▲' ? ' up' : mark === ' ▼' ? ' down' : ''}` }, format(key, value) + mark);
            });
            return h('tr', {}, h('td', {}, `${tagName(tag)}（${tag}）`), h('td', { class: 'num' }, String(report.tag_cases?.[tag] ?? '')), ...cells);
        });
        return h(
            'section',
            { class: 'card' },
            h('h2', {}, '切り口ごとの成績'),
            h('p', { class: 'lead' }, '▲ は全体より 0.05 以上良い、▼ は悪い（CERは小さいほど良い）。件数が少ない切り口は偶然の差かもしれません。'),
            h('table', {}, h('thead', {}, head), h('tbody', {}, overall, ...body)),
        );
    }

    function confusionCard(report: Report): HTMLElement {
        const section = h('section', { class: 'card' }, h('h2', {}, '混同行列（何を何と間違えたか）'));
        if (report.confusion === null) {
            section.append(h('p', { class: 'note' }, 'このモデルはラベルを出さないので、対象外です。'));
            return section;
        }
        const { labels, matrix } = report.confusion;
        // 正解にも予測にも1度も出てこないラベルは省く
        const used = labels.map((_, i) => i).filter((i) => matrix[i].some((n) => n > 0) || matrix.some((row) => row[i] > 0));
        if (used.length === 0) {
            section.append(h('p', { class: 'note' }, '比べられた組がありません。'));
            return section;
        }
        const head = h('tr', {}, h('th', {}, '正解 ＼ 予測'), ...used.map((j) => h('th', { class: 'rot' }, labels[j])), h('th', { class: 'num' }, '正解率'));
        const body = used.map((i) => {
            const total = matrix[i].reduce((a, b) => a + b, 0);
            const cells = used.map((j) => {
                const n = matrix[i][j];
                const share = total > 0 ? n / total : 0;
                const step = n === 0 ? -1 : Math.min(RAMP.length - 1, Math.floor(share * RAMP.length));
                const td = h('td', { class: 'cell', style: step < 0 ? '' : `background: ${RAMP[step]}; color: ${step >= 3 ? '#fff' : 'var(--text)'}` }, n === 0 ? '' : String(n));
                if (n > 0) {
                    showTooltip(td, `正解 ${labels[i]} を ${labels[j]} と予測: ${n}件（正解 ${labels[i]} の ${(share * 100).toFixed(0)}%）`);
                }
                return td;
            });
            const acc = total > 0 ? matrix[i][i] / total : null;
            return h('tr', {}, h('th', {}, labels[i]), ...cells, h('td', { class: 'num' }, acc === null ? '—' : `${(acc * 100).toFixed(0)}%`));
        });
        section.append(
            h('p', { class: 'lead' }, '行が正解、列が予測。対角線（左上から右下）が正しく当てた数。色の濃さは、その行（正解のラベル）の中での割合です。'),
            h('table', { class: 'confusion' }, h('thead', {}, head), h('tbody', {}, ...body)),
        );
        return section;
    }

    function casesCard(report: Report): HTMLElement {
        const keys = ['f1', 'label_acc', 'cer', 'deco_f1'];
        const head = h('tr', {}, h('th', {}, 'ケース'), h('th', {}, '切り口'), ...keys.map((k) => h('th', { class: 'num' }, METRIC_BY_KEY.get(k)!.name)), h('th', { class: 'num' }, '秒'), h('th', {}, 'エラー'));
        const body = report.cases.map((c) => {
            const tags = Object.entries(c.tags).map(([k, v]) => tagName(`${k}=${v === true ? 'True' : v === false ? 'False' : v}`)).join('・');
            const tr = h(
                'tr',
                { class: 'clickable' },
                h('td', {}, c.id),
                h('td', { class: 'meta' }, tags),
                ...keys.map((k) => h('td', { class: `num${c.scores[k] == null ? ' na' : ''}` }, format(k, c.scores[k]))),
                h('td', { class: 'num' }, c.seconds.toFixed(2)),
                h('td', { class: 'down' }, c.error ?? ''),
            );
            tr.addEventListener('click', () => {
                location.hash = `#/run/${report.run_id}/case/${c.id}`;
            });
            return tr;
        });
        return h(
            'section',
            { class: 'card' },
            h('h2', {}, 'ケース一覧'),
            h('p', { class: 'lead' }, '行をクリックすると、ページの上に正解と予測の枠を重ねて見られます。'),
            h('table', {}, h('thead', {}, head), h('tbody', {}, ...body)),
        );
    }

    // ---------- ケース詳細 ----------

    // ai/evaluation/metrics.py の iou / match と同じ計算（ページの上で、どの予測がどの正解に当たったかを色分けするため）
    function iou(a: number[], b: number[]): number {
        const area = (x: number[]) => Math.max(x[2] - x[0], 0) * Math.max(x[3] - x[1], 0);
        const inter = area([Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])]);
        const union = area(a) + area(b) - inter;
        return union > 0 ? inter / union : 0;
    }

    function match(preds: IrElement[], gts: IrElement[], threshold: number): [number, number, number][] {
        const candidates: [number, number, number][] = [];
        preds.forEach((p, i) => gts.forEach((g, j) => {
            const score = iou(p.bbox, g.bbox);
            if (score >= threshold) {
                candidates.push([score, i, j]);
            }
        }));
        // IoU が大きい順（同じなら番号の大きい順。Python の sort(reverse=True) と同じ順）
        candidates.sort((x, y) => y[0] - x[0] || y[1] - x[1] || y[2] - x[2]);
        const usedP = new Set<number>();
        const usedG = new Set<number>();
        const pairs: [number, number, number][] = [];
        for (const [score, i, j] of candidates) {
            if (usedP.has(i) || usedG.has(j)) {
                continue;
            }
            pairs.push([i, j, score]);
            usedP.add(i);
            usedG.add(j);
        }
        return pairs;
    }

    /** 文字列を、装飾（runs）を付けて表示する */
    function richText(element: IrElement | undefined): Node {
        if (element === undefined) {
            return h('span', { class: 'na' }, '（なし）');
        }
        if (!element.runs) {
            return document.createTextNode(element.text ?? '');
        }
        const span = h('span');
        for (const run of element.runs) {
            let node: Node = document.createTextNode(run.text);
            if (run.bold) node = h('b', {}, node);
            if (run.italic) node = h('i', {}, node);
            if (run.underline) node = h('u', {}, node);
            span.append(node);
        }
        return span;
    }

    function labelText(element: IrElement | undefined): string {
        if (element === undefined) {
            return '—';
        }
        const label = element.label ?? `（ラベルなし: ${element.kind}）`;
        return element.level ? `${label}（レベル ${element.level}）` : label;
    }

    async function renderCase(runId: string, caseId: string): Promise<void> {
        const report = await getJson<Report>(`/runs/${runId}/report.json`);
        const caseReport = report.cases.find((c) => c.id === caseId);
        const [gt, pred] = await Promise.all([
            getJson<IrDocument>(`/datasets/${report.dataset}/cases/${caseId}/gt.labeled.json`),
            getJson<IrDocument>(`/runs/${runId}/preds/${caseId}.labeled.json`),
        ]);

        const showGt = h('input', { type: 'checkbox', checked: '' });
        const showPred = h('input', { type: 'checkbox', checked: '' });
        const detail = h('section', { class: 'card detail' }, h('h2', {}, 'ブロックの比較'), h('p', { class: 'note' }, 'ページの枠をクリックすると、正解と予測を並べて表示します。'));
        const pagesBox = h('div');
        const boxes: HTMLElement[] = [];

        const predPages = new Map(pred.pages.map((p) => [p.page, p]));
        for (const gtPage of gt.pages) {
            const predElements = predPages.get(gtPage.page)?.elements ?? [];
            const pairs = match(predElements, gtPage.elements, report.iou_threshold);
            const gtPair = new Map(pairs.map(([i, j, score]) => [j, { i, score }]));
            const predPair = new Map(pairs.map(([i, j, score]) => [i, { j, score }]));

            const view = h('div', { class: 'page-view' });
            const img = h('img', {
                src: `${API}/datasets/${report.dataset}/cases/${caseId}/page-${gtPage.page}.png`,
                alt: `${gtPage.page}ページ目`,
            });
            img.addEventListener('error', () => {
                // 古いデータセットには画像が無い。ページの縦横比の白い四角にして、枠だけ重ねる
                img.replaceWith(h(
                    'div',
                    { class: 'page-missing', style: `aspect-ratio: ${gtPage.width} / ${gtPage.height}` },
                    `ページの画像がありません。ai/ で python -m evaluation.images evaluation/data/${report.dataset} を実行すると表示されます。`,
                ));
            });
            view.append(img);

            const select = (gtIndex: number | undefined, predIndex: number | undefined, box: HTMLElement) => {
                boxes.forEach((b) => b.classList.remove('selected'));
                box.classList.add('selected');
                const g = gtIndex === undefined ? undefined : gtPage.elements[gtIndex];
                const p = predIndex === undefined ? undefined : predElements[predIndex];
                const score = gtIndex !== undefined ? gtPair.get(gtIndex)?.score : predIndex !== undefined ? predPair.get(predIndex)?.score : undefined;
                const verdict = g && p ? '当たり（正解と予測が対応した）' : g ? '見逃し（正解に対応する予測が無い）' : '誤検出（正解に無いブロックを出した）';
                const labelOk = g && p && p.label !== undefined ? (g.label === p.label ? '✓ 一致' : '✗ 不一致') : '';
                detail.replaceChildren(
                    h('h2', {}, 'ブロックの比較'),
                    h(
                        'dl',
                        {},
                        h('dt', {}, '判定'), h('dd', {}, verdict),
                        h('dt', {}, 'IoU'), h('dd', {}, score === undefined ? '—' : score.toFixed(3)),
                        h('dt', {}, '正解のラベル'), h('dd', {}, labelText(g)),
                        h('dt', {}, '予測のラベル'), h('dd', {}, `${labelText(p)} ${labelOk}`),
                        h('dt', {}, '正解の文字'), h('dd', {}, richText(g)),
                        h('dt', {}, '予測の文字'), h('dd', {}, richText(p)),
                    ),
                );
            };

            const place = (element: IrElement, cls: string, onClick: (box: HTMLElement) => void, tip: string) => {
                const [x0, y0, x1, y1] = element.bbox;
                // bbox はページの幅・高さで割った 0〜1 の値なので、% にするだけで画像の大きさに関係なく重なる
                const box = h('div', {
                    class: `box ${cls}`,
                    style: `left: ${x0 * 100}%; top: ${y0 * 100}%; width: ${(x1 - x0) * 100}%; height: ${(y1 - y0) * 100}%`,
                });
                box.addEventListener('click', (e) => {
                    e.stopPropagation();
                    onClick(box);
                });
                showTooltip(box, tip);
                boxes.push(box);
                view.append(box);
            };

            gtPage.elements.forEach((g, j) => {
                const pair = gtPair.get(j);
                place(g, `gt ${pair ? 'gt-hit' : 'gt-miss'}`, (box) => select(j, pair?.i, box), `正解: ${labelText(g)}${pair ? '' : '（見逃し）'}`);
            });
            predElements.forEach((p, i) => {
                const pair = predPair.get(i);
                place(p, `pred ${pair ? 'pred-hit' : 'pred-fp'}`, (box) => select(pair?.j, i, box), `予測: ${labelText(p)}${pair ? '' : '（誤検出）'}`);
            });
            pagesBox.append(h('p', { class: 'meta' }, `${gtPage.page}ページ目`), view);
        }

        const toggle = () => {
            for (const box of boxes) {
                const hidden = (box.classList.contains('gt') && !showGt.checked) || (box.classList.contains('pred') && !showPred.checked);
                box.hidden = hidden;
            }
        };
        showGt.addEventListener('change', toggle);
        showPred.addEventListener('change', toggle);

        const legend = h(
            'div',
            { class: 'legend' },
            h('span', {}, h('span', { class: 'swatch', style: 'border: 2px solid var(--series-1)' }), '予測（当たり）'),
            h('span', {}, h('span', { class: 'swatch', style: 'border: 2px solid var(--series-2); background: rgba(235,104,52,0.08)' }), '予測（誤検出）'),
            h('span', {}, h('span', { class: 'swatch', style: 'border: 2px dashed var(--series-3); background: rgba(27,175,122,0.08)' }), '正解（見逃し）'),
            h('span', {}, h('span', { class: 'swatch', style: 'border: 1px dashed var(--muted)' }), '正解（当たり）'),
        );
        const scores = caseReport
            ? ['f1', 'label_acc', 'cer', 'deco_f1'].map((k) => `${METRIC_BY_KEY.get(k)!.name} ${format(k, caseReport.scores[k])}`).join(' ／ ')
            : '';

        app.replaceChildren(
            h(
                'section',
                { class: 'card' },
                h('div', { class: 'breadcrumb' }, h('a', { href: '#/' }, '結果一覧'), ' › ', h('a', { href: `#/run/${runId}` }, `${report.model.name} × ${report.dataset}`)),
                h('h2', {}, `ケース ${caseId}`),
                h('p', { class: 'meta' }, scores),
                caseReport?.error ? h('div', { class: 'error-box' }, `モデルが失敗しました: ${caseReport.error}`) : null,
            ),
            h(
                'div',
                { class: 'case-layout' },
                h('section', { class: 'card' }, legend, h('div', { class: 'toggles' }, h('label', {}, showGt, ' 正解を表示'), h('label', {}, showPred, ' 予測を表示')), pagesBox),
                detail,
            ),
        );
    }

    // ---------- 画面の切り替え ----------

    async function route(): Promise<void> {
        tooltip.hidden = true;
        app.replaceChildren(h('p', { class: 'meta' }, '読み込み中…'));
        const m = location.hash.match(/^#\/run\/([\w-]+)(?:\/case\/(\d{4}))?$/);
        try {
            if (m && m[2]) {
                await renderCase(m[1], m[2]);
            } else if (m) {
                await renderRun(m[1]);
            } else {
                await renderHome();
            }
        } catch (err) {
            app.replaceChildren(h('section', { class: 'card' }, errorBox(err), h('p', {}, h('a', { href: '#/' }, '← 結果一覧'))));
        }
        window.scrollTo(0, 0);
    }

    window.addEventListener('hashchange', route);
    route();
})();
