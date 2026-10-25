type AddKind = 'heading1' | 'heading2' | 'heading3' | 'paragraph' | 'list' | 'math' | 'title' | 'toc' | 'table' | 'image';

function createBlock(kind: AddKind): Block {
  const id = generateId();
  switch (kind) {
    case 'heading1':
    case 'heading2':
    case 'heading3':
      return { id, type: 'heading', level: Number(kind.slice(-1)) as HeadingLevel, text: '' };
    case 'paragraph':
      return { id, type: 'paragraph', content: [] };
    case 'list':
      return { id, type: 'list', marker: DEFAULT_LIST_MARKER, items: [{ content: [] }] };
    case 'math':
      return { id, type: 'math', displayMode: true, tex: '' };
    case 'title':
      return { id, type: 'title', text: '', author: '', date: '' };
    case 'toc':
      return { id, type: 'toc' };
    case 'table':
      return {
        id,
        type: 'table',
        columns: [{ align: 'l' }, { align: 'l' }],
        header: { cells: ['', ''] },
        rows: [{ cells: ['', ''] }, { cells: ['', ''] }],
      };
    case 'image':
      return { id, type: 'image', src: '', widthRatio: DEFAULT_IMAGE_WIDTH_RATIO };
  }
}

const DEFAULT_IMAGE_WIDTH_RATIO = 0.6;
/** dvipdfmx の graphicx で読める拡張子 */
const TEX_IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'pdf'];

function createEmptyDocument(title = ''): BlockDocument {
  return { schemaVersion: SCHEMA_VERSION, id: generateId(), title, blocks: [] };
}

class BlockEditor {
  private doc: BlockDocument = createEmptyDocument();
  /** 直近でフォーカスしたブロック。新規ブロックはこの後ろに挿入する。 */
  private selectedId: BlockId | null = null;
  /** 画像プレビュー用のURL。保存対象ではないのでモデルには入れない。 */
  private readonly imageUrls = new Map<BlockId, string>();

  constructor(
    private readonly listEl: HTMLElement,
    private readonly onChange: (doc: BlockDocument) => void,
  ) {
    this.listEl.addEventListener('focusin', (e) => {
      const el = (e.target as HTMLElement).closest<HTMLElement>('.block');
      if (el) this.select(el.dataset.id ?? null);
    });
  }

  load(doc: BlockDocument): void {
    this.doc = doc;
    this.selectedId = null;
    for (const id of [...this.imageUrls.keys()]) this.revokeImageUrl(id);
    this.render();
    this.changed();
  }

  /** 文書が変わったときの共通処理。目次の一覧を更新してからTeXを再生成する。 */
  private changed(): void {
    this.listEl.querySelectorAll<HTMLElement>('.toc-list').forEach((el) => this.fillToc(el));
    this.onChange(this.doc);
  }

  /** 見出しブロックから目次の一覧を作る。番号は jsarticle の section / subsection / subsubsection と同じ規則。 */
  private fillToc(el: HTMLElement): void {
    const counters = [0, 0, 0];
    const items: HTMLElement[] = [];
    for (const block of this.doc.blocks) {
      if (block.type !== 'heading') continue;
      counters[block.level - 1] += 1;
      counters.fill(0, block.level);
      const row = document.createElement('div');
      row.className = `toc-row toc-level-${block.level}`;
      row.textContent = `${counters.slice(0, block.level).join('.')}  ${block.text || '(無題)'}`;
      items.push(row);
    }
    if (items.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'toc-empty';
      empty.textContent = '見出しがありません';
      items.push(empty);
    }
    el.replaceChildren(...items);
  }

  add(kind: AddKind): void {
    if (kind === 'toc' || kind === 'title') {
      // 目次とタイトルは1文書に1つだけ。既にあればそこへ移動する。
      const existing = this.doc.blocks.find((b) => b.type === kind);
      if (existing) {
        const el = this.listEl.querySelector<HTMLElement>(`[data-id="${existing.id}"]`);
        el?.scrollIntoView({ block: 'center' });
        this.select(existing.id);
        return;
      }
    }
    const block = createBlock(kind);
    const index = this.doc.blocks.findIndex((b) => b.id === this.selectedId);
    // タイトルは常に先頭。それ以外は選択中のブロックの後ろ（未選択なら末尾）
    const at = kind === 'title' ? 0 : index < 0 ? this.doc.blocks.length : index + 1;
    this.doc.blocks.splice(at, 0, block);
    this.selectedId = block.id;
    this.render();
    this.changed();
    this.listEl.querySelector<HTMLElement>(`[data-id="${block.id}"] .block-input`)?.focus();
  }

  private move(id: BlockId, delta: -1 | 1): void {
    const blocks = this.doc.blocks;
    const from = blocks.findIndex((b) => b.id === id);
    const to = from + delta;
    if (from < 0 || to < 0 || to >= blocks.length) return;
    // タイトルは先頭に固定し、他のブロックもタイトルより上へは動かさない
    if (blocks[from].type === 'title' || blocks[to].type === 'title') return;
    [blocks[from], blocks[to]] = [blocks[to], blocks[from]];
    this.render();
    this.changed();
  }

  private remove(id: BlockId): void {
    this.doc.blocks = this.doc.blocks.filter((b) => b.id !== id);
    if (this.selectedId === id) this.selectedId = null;
    this.revokeImageUrl(id);
    this.render();
    this.changed();
  }

  private revokeImageUrl(id: BlockId): void {
    const url = this.imageUrls.get(id);
    if (url) URL.revokeObjectURL(url);
    this.imageUrls.delete(id);
  }

  private select(id: BlockId | null): void {
    this.selectedId = id;
    for (const el of this.listEl.querySelectorAll<HTMLElement>('.block')) {
      el.classList.toggle('selected', el.dataset.id === id);
    }
  }

  private render(): void {
    this.listEl.replaceChildren();
    if (this.doc.blocks.length === 0) {
      const hint = document.createElement('p');
      hint.className = 'empty-hint';
      hint.textContent = '左のパレットから見出しや段落を追加してください';
      this.listEl.append(hint);
      return;
    }
    for (const block of this.doc.blocks) {
      this.listEl.append(this.renderBlock(block));
    }
    this.listEl.querySelectorAll<HTMLElement>('.toc-list').forEach((el) => this.fillToc(el));
    this.select(this.selectedId);
  }

  private renderBlock(block: Block): HTMLElement {
    const el = document.createElement('div');
    el.className = `block block-${block.type}`;
    el.dataset.id = block.id;

    const controls = document.createElement('div');
    controls.className = 'block-controls';
    if (block.type !== 'title') {
      controls.append(
        this.controlButton('↑', '上へ', () => this.move(block.id, -1)),
        this.controlButton('↓', '下へ', () => this.move(block.id, 1)),
      );
    }
    controls.append(this.controlButton('✕', '削除', () => this.remove(block.id)));

    el.append(controls, this.renderInput(block));
    return el;
  }

  private controlButton(label: string, title: string, onClick: () => void): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.title = title;
    button.addEventListener('click', onClick);
    return button;
  }

  private renderInput(block: Block): HTMLElement {
    switch (block.type) {
      case 'title': {
        const wrap = document.createElement('div');
        wrap.className = 'title-wrap';
        const field = (value: string, placeholder: string, cls: string, apply: (v: string) => void) => {
          const input = document.createElement('input');
          input.type = 'text';
          input.className = `block-input ${cls}`;
          input.placeholder = placeholder;
          input.value = value;
          input.addEventListener('input', () => {
            apply(input.value);
            this.changed();
          });
          return input;
        };
        const row = document.createElement('div');
        row.className = 'title-row';
        row.append(
          field(block.author, '著者', 'title-sub', (v) => (block.author = v)),
          field(block.date, '日付', 'title-sub', (v) => (block.date = v)),
        );
        wrap.append(field(block.text, 'タイトル', 'title-main', (v) => (block.text = v)), row);
        return wrap;
      }
      case 'heading': {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = `block-input heading-${block.level}`;
        input.placeholder = `見出し${block.level}`;
        input.value = block.text;
        input.addEventListener('input', () => {
          block.text = input.value;
          this.changed();
        });
        return input;
      }
      case 'paragraph':
        return createRichTextEditor({
          content: block.content,
          singleLine: false,
          placeholder: '本文を入力',
          onChange: (content) => {
            block.content = content;
            this.changed();
          },
        });
      case 'list':
        return this.renderList(block);
      case 'math': {
        const wrap = document.createElement('div');
        wrap.className = 'math-wrap';
        const preview = document.createElement('div');
        preview.className = 'math-preview';
        const update = () => this.renderMathPreview(preview, block);
        const area = this.textarea(block.tex, 'TeX数式 例: E = mc^2', (v) => {
          block.tex = v;
          update();
        });
        area.classList.add('math-input');
        const label = document.createElement('label');
        label.className = 'math-mode';
        const check = document.createElement('input');
        check.type = 'checkbox';
        check.checked = block.displayMode;
        check.addEventListener('change', () => {
          block.displayMode = check.checked;
          update();
          this.changed();
        });
        label.append(check, ' 別行立て');
        wrap.append(area, preview, label);
        update();
        return wrap;
      }
      case 'toc': {
        const wrap = document.createElement('div');
        wrap.className = 'toc-wrap';
        const title = document.createElement('div');
        title.className = 'toc-title';
        title.textContent = '目次';
        const list = document.createElement('div');
        list.className = 'toc-list';
        wrap.append(title, list);
        return wrap;
      }
      case 'table':
        return this.renderTable(block);
      case 'image':
        return this.renderImage(block);
    }
  }

  private renderTable(block: TableBlock): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';

    const grid = document.createElement('div');
    grid.className = 'table-grid';

    const toolbar = document.createElement('div');
    toolbar.className = 'table-tools';

    // 行・列の増減のときだけグリッドを作り直す。セルの入力中は作り直さない（フォーカスを保つ）
    const build = (focus?: { row: number; col: number }) => {
      const cols = block.columns.length;
      grid.style.gridTemplateColumns = `repeat(${cols}, minmax(80px, 1fr))`;
      grid.replaceChildren();

      for (const [c, column] of block.columns.entries()) {
        const select = document.createElement('select');
        select.className = 'table-align';
        select.title = '列の配置';
        select.append(new Option('左', 'l'), new Option('中', 'c'), new Option('右', 'r'));
        select.value = column.align;
        select.addEventListener('change', () => {
          column.align = select.value as TableCellAlign;
          this.changed();
        });
        const del = this.controlButton('✕', '列を削除', () => {
          if (block.columns.length <= 1) return;
          block.columns.splice(c, 1);
          for (const row of [block.header, ...block.rows]) row?.cells.splice(c, 1);
          build();
          this.changed();
        });
        del.classList.add('table-del');
        const head = document.createElement('div');
        head.className = 'table-colhead';
        head.append(select, del);
        grid.append(head);
      }

      const rows = block.header ? [block.header, ...block.rows] : block.rows;
      rows.forEach((row, r) => {
        row.cells.forEach((text, c) => {
          const input = document.createElement('input');
          input.type = 'text';
          input.className = 'block-input table-cell';
          if (block.header && r === 0) input.classList.add('table-header-cell');
          input.value = text;
          input.addEventListener('input', () => {
            row.cells[c] = input.value;
            this.changed();
          });
          grid.append(input);
        });
      });

      if (focus) {
        grid.querySelectorAll<HTMLElement>('.table-cell')[focus.row * cols + focus.col]?.focus();
      }
    };

    const newRow = (): TableRow => ({ cells: block.columns.map(() => '') });

    const addRow = this.controlButton('＋行', '行を追加', () => {
      block.rows.push(newRow());
      build({ row: block.rows.length - 1 + (block.header ? 1 : 0), col: 0 });
      this.changed();
    });
    const delRow = this.controlButton('－行', '最後の行を削除', () => {
      if (block.rows.length <= 1) return;
      block.rows.pop();
      build();
      this.changed();
    });
    const addCol = this.controlButton('＋列', '列を追加', () => {
      block.columns.push({ align: 'l' });
      for (const row of [block.header, ...block.rows]) row?.cells.push('');
      build();
      this.changed();
    });

    const headerLabel = document.createElement('label');
    const headerCheck = document.createElement('input');
    headerCheck.type = 'checkbox';
    headerCheck.checked = block.header !== undefined;
    headerCheck.addEventListener('change', () => {
      block.header = headerCheck.checked ? newRow() : undefined;
      build();
      this.changed();
    });
    headerLabel.append(headerCheck, ' 見出し行');

    toolbar.append(addRow, delRow, addCol, headerLabel);

    const caption = document.createElement('input');
    caption.type = 'text';
    caption.className = 'caption-input';
    caption.placeholder = 'キャプション（省略可）';
    caption.value = block.caption ?? '';
    caption.addEventListener('input', () => {
      block.caption = caption.value === '' ? undefined : caption.value;
      this.changed();
    });

    build();
    wrap.append(grid, toolbar, caption);
    return wrap;
  }

  private renderImage(block: ImageBlock): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'image-wrap';

    // 座標はすべて本文幅に対する比率。offsetX / offsetY がどちらも未指定なら中央寄せ。
    const width = () => block.widthRatio ?? 1;
    const isPlaced = () => block.offsetX !== undefined || block.offsetY !== undefined;
    const posX = () => block.offsetX ?? (1 - width()) / 2;
    const posY = () => block.offsetY ?? 0;
    const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));
    const snap = (v: number) => Math.round(v * 200) / 200; // 0.005刻み

    const stage = document.createElement('div');
    stage.className = 'image-stage';
    const box = document.createElement('div');
    box.className = 'image-box';
    box.tabIndex = 0; // フォーカスでブロックを選択状態にするため
    const preview = document.createElement('img');
    preview.className = 'image-preview';
    preview.alt = block.caption ?? '';
    preview.draggable = false;
    const handle = document.createElement('div');
    handle.className = 'image-handle';
    handle.title = 'ドラッグで拡大縮小';
    box.append(preview, handle);
    stage.append(box);

    const note = document.createElement('p');
    note.className = 'image-note';

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.hidden = true;

    const pick = document.createElement('button');
    pick.type = 'button';
    pick.className = 'image-pick';
    pick.addEventListener('click', () => fileInput.click());

    const range = document.createElement('input');
    range.type = 'range';
    range.min = '10';
    range.max = '100';
    range.step = '5';
    const rangeText = document.createElement('span');

    /** モデルの値をステージ上のボックスの位置・大きさに反映する（DOMは作り直さない） */
    const layout = () => {
      const w = stage.clientWidth;
      box.style.left = `${posX() * 100}%`;
      box.style.width = `${width() * 100}%`;
      box.style.top = `${posY() * w}px`;
      // ドラッグでさらに下へ動かせるよう、画像の下に少し余白を足す
      stage.style.height = `${posY() * w + box.offsetHeight + 24}px`;
      range.value = String(Math.round(width() * 100));
      rangeText.textContent = ` 幅 ${range.value}%`;
    };
    new ResizeObserver(layout).observe(stage);
    preview.addEventListener('load', layout);

    const refresh = () => {
      const url = this.imageUrls.get(block.id);
      stage.hidden = !url;
      if (url) preview.src = url;
      pick.textContent = block.src ? `${block.src}（変更）` : '画像を選択';
      const ext = block.src.split('.').pop()?.toLowerCase() ?? '';
      if (!block.src) {
        note.textContent = '';
      } else if (!TEX_IMAGE_EXTENSIONS.includes(ext)) {
        note.textContent = '⚠ dvipdfmx では読めない形式です（png / jpg / pdf に変換してください）';
      } else {
        note.textContent = `TeXをコンパイルするときは、${block.src} を .tex と同じフォルダに置いてください`;
      }
      layout();
    };

    /** ポインタのドラッグを開始する。onMove には開始位置からの移動量（本文幅に対する比率）を渡す。 */
    const startDrag = (e: PointerEvent, target: HTMLElement, onMove: (dx: number, dy: number) => void) => {
      e.preventDefault();
      e.stopPropagation();
      const w = stage.clientWidth;
      if (w === 0) return;
      const startX = e.clientX;
      const startY = e.clientY;
      target.setPointerCapture(e.pointerId);
      const move = (ev: PointerEvent) => {
        onMove((ev.clientX - startX) / w, (ev.clientY - startY) / w);
        layout();
        this.changed();
      };
      const end = () => {
        target.removeEventListener('pointermove', move);
        target.removeEventListener('pointerup', end);
        target.removeEventListener('pointercancel', end);
      };
      target.addEventListener('pointermove', move);
      target.addEventListener('pointerup', end);
      target.addEventListener('pointercancel', end);
    };

    box.addEventListener('pointerdown', (e) => {
      const sx = posX();
      const sy = posY();
      startDrag(e, box, (dx, dy) => {
        block.offsetX = snap(clamp(sx + dx, 0, 1 - width()));
        block.offsetY = snap(clamp(sy + dy, 0, 1));
      });
      box.focus();
    });

    handle.addEventListener('pointerdown', (e) => {
      // 拡大縮小は左上を固定するので、中央寄せのままだと位置が動いてしまう。先に位置を確定させる。
      block.offsetX = posX();
      block.offsetY = posY();
      const sw = width();
      startDrag(e, handle, (dx) => {
        block.widthRatio = snap(clamp(sw + dx, 0.1, 1 - posX()));
      });
    });

    range.addEventListener('input', () => {
      block.widthRatio = Number(range.value) / 100;
      if (block.offsetX !== undefined) block.offsetX = snap(Math.min(block.offsetX, 1 - width()));
      layout();
      this.changed();
    });

    const align = (label: string, title: string, x: (w: number) => number) => {
      const button = this.controlButton(label, title, () => {
        const sx = x(width());
        const sy = posY();
        if (sy === 0 && sx === (1 - width()) / 2) {
          // 中央・上端に戻すなら、配置モードをやめて通常の中央寄せにする
          block.offsetX = undefined;
          block.offsetY = undefined;
        } else {
          block.offsetX = snap(sx);
          block.offsetY = sy;
        }
        layout();
        this.changed();
      });
      button.classList.add('image-align');
      return button;
    };

    fileInput.addEventListener('change', () => {
      const file = fileInput.files?.[0];
      if (!file) return;
      this.revokeImageUrl(block.id);
      this.imageUrls.set(block.id, URL.createObjectURL(file));
      block.src = file.name;
      refresh();
      this.changed();
    });

    const controls = document.createElement('div');
    controls.className = 'image-controls';
    const rangeLabel = document.createElement('label');
    rangeLabel.className = 'image-width';
    rangeLabel.append(range, rangeText);
    controls.append(
      align('左', '左に寄せる', () => 0),
      align('中央', '中央に寄せる', (w) => (1 - w) / 2),
      align('右', '右に寄せる', (w) => 1 - w),
      rangeLabel,
    );

    const caption = document.createElement('input');
    caption.type = 'text';
    caption.className = 'caption-input';
    caption.placeholder = 'キャプション（省略可）';
    caption.value = block.caption ?? '';
    caption.addEventListener('input', () => {
      block.caption = caption.value === '' ? undefined : caption.value;
      preview.alt = caption.value;
      this.changed();
    });

    refresh();
    wrap.append(pick, fileInput, stage, note, controls, caption);
    return wrap;
  }

  private renderMathPreview(el: HTMLElement, block: MathBlock): void {
    el.replaceChildren();
    if (typeof katex === 'undefined' || block.tex.trim() === '') return;
    // KaTeX は align* を解釈しないので、別行立ては aligned で包んで描画する（TeX出力は変えない）
    const tex = block.displayMode ? `\\begin{aligned}${block.tex}\\end{aligned}` : block.tex;
    katex.render(tex, el, { displayMode: block.displayMode, throwOnError: false });
  }

  private renderList(block: ListBlock): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'list-wrap';

    const select = document.createElement('select');
    select.className = 'list-marker-select';
    select.title = 'マーカーの種類';
    for (const [value, spec] of Object.entries(LIST_MARKERS)) {
      select.append(new Option(spec.label, value));
    }
    select.value = block.marker ?? DEFAULT_LIST_MARKER;

    const rows = document.createElement('div');
    rows.className = 'list-items';

    const glyphs = () => {
      const spec = LIST_MARKERS[block.marker ?? DEFAULT_LIST_MARKER];
      rows.querySelectorAll<HTMLElement>('.list-glyph').forEach((el, i) => {
        el.textContent = spec.glyph(i + 1);
      });
    };

    // 項目の増減のときだけ行を作り直す。入力中は作り直さない（フォーカスを保つため）
    const build = (focusIndex?: number) => {
      rows.replaceChildren();
      block.items.forEach((item, i) => {
        const row = document.createElement('div');
        row.className = 'list-row';
        const glyph = document.createElement('span');
        glyph.className = 'list-glyph';
        const input = createRichTextEditor({
          content: item.content,
          singleLine: true,
          placeholder: '項目',
          onChange: (content) => {
            item.content = content;
            this.changed();
          },
        });
        input.classList.add('list-item');
        input.addEventListener('keydown', (e) => {
          if (e.isComposing) return; // 日本語入力の変換確定のEnterは無視する
          if (e.key === 'Enter') {
            e.preventDefault();
            block.items.splice(i + 1, 0, { content: [] });
            build(i + 1);
            this.changed();
          } else if (e.key === 'Backspace' && item.content.length === 0 && block.items.length > 1) {
            e.preventDefault();
            block.items.splice(i, 1);
            build(Math.max(0, i - 1));
            this.changed();
          }
        });
        row.append(glyph, input);
        rows.append(row);
      });
      glyphs();
      if (focusIndex !== undefined) {
        rows.querySelectorAll<HTMLElement>('.list-item')[focusIndex]?.focus();
      }
    };

    select.addEventListener('change', () => {
      block.marker = select.value as ListMarker;
      glyphs();
      this.changed();
    });

    build();
    wrap.append(select, rows);
    return wrap;
  }

  private textarea(value: string, placeholder: string, apply: (value: string) => void): HTMLTextAreaElement {
    const area = document.createElement('textarea');
    area.className = 'block-input';
    area.placeholder = placeholder;
    area.value = value;
    area.rows = 1;
    const fit = () => {
      area.style.height = 'auto';
      area.style.height = `${area.scrollHeight}px`;
    };
    area.addEventListener('input', () => {
      apply(area.value);
      fit();
      this.changed();
    });
    // 描画後でないと scrollHeight が確定しない
    requestAnimationFrame(fit);
    return area;
  }
}
