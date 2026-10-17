// KaTeX は CDN から読む。オフラインなどで読めない場合は undefined になるので、使う前に typeof で確認する。
declare const katex:
  | { render(tex: string, el: HTMLElement, options: { displayMode: boolean; throwOnError: boolean }): void }
  | undefined;

/** カーソルをチップの前後に置けるよう、チップの両隣に入れておく幅ゼロの文字。保存時には取り除く。 */
const ZWSP = '​';

type TextSegment = Extract<InlineSegment, { type: 'text' }>;

/** 文中に置く小さな部品（チップ）の種類。クリックするとその場で中身を編集できる */
type ChipKind = 'math' | 'footnote';

/** 文字の装飾のうち、オン・オフで切り替えるもの。document.execCommand のコマンド名と同じにしている */
type FormatKind = 'bold' | 'italic' | 'underline';
const FORMAT_KINDS: FormatKind[] = ['bold', 'italic', 'underline'];

/** 装飾のうち有効なものだけを取り出す（false や空のURLのキーは作らない） */
function marksOf(seg: TextMarks): TextMarks {
  const marks: TextMarks = {};
  for (const kind of FORMAT_KINDS) if (seg[kind]) marks[kind] = true;
  if (seg.href) marks.href = seg.href;
  return marks;
}

function sameMarks(a: TextMarks, b: TextMarks): boolean {
  return FORMAT_KINDS.every((kind) => !a[kind] === !b[kind]) && (a.href ?? '') === (b.href ?? '');
}

/** 装飾が同じで隣り合う text を結合し、空の text を捨てる */
function normalizeRich(content: RichText): RichText {
  const out: RichText = [];
  for (const seg of content) {
    const last = out[out.length - 1];
    if (seg.type === 'text') {
      if (seg.text === '') continue;
      const marks = marksOf(seg);
      if (last?.type === 'text' && sameMarks(last, marks)) last.text += seg.text;
      else out.push({ type: 'text', text: seg.text, ...marks });
    } else if (seg.type === 'math') {
      out.push({ type: 'math', tex: seg.tex });
    } else {
      out.push({ type: 'footnote', text: seg.text });
    }
  }
  return out;
}

/** リンクに使えるURLの形に整える。使えないものは null */
function normalizeLinkUrl(raw: string): string | null {
  const url = raw.trim();
  if (url === '') return null;
  if (/^(https?:\/\/|mailto:)/i.test(url)) return url;
  // javascript: など、http(s) と mailto 以外の書き方は受け付けない
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return null;
  if (/^[^\s@/]+@[^\s@]+$/.test(url)) return `mailto:${url}`;
  return `https://${url}`;
}

interface RichTextOptions {
  content: RichText;
  /** true: 改行を入れられない（箇条書きの項目用）。Enter の扱いは呼び出し側に任せる */
  singleLine: boolean;
  placeholder: string;
  onChange: (content: RichText) => void;
}

interface RichTextCommands {
  el: HTMLElement;
  insertChip: (kind: ChipKind) => void;
  format: (kind: FormatKind) => void;
  editLink: () => void;
}

/** 直近でフォーカスしたリッチテキスト部品。パレットのボタンやショートカットから操作するために覚えておく。 */
let activeRich: RichTextCommands | null = null;

/** フォーカス中のリッチテキスト部品があれば fn を実行する。できなければ false。 */
function withActiveRich(fn: (rich: RichTextCommands) => void): boolean {
  const active = activeRich;
  if (!active || !active.el.isConnected || !active.el.contains(document.activeElement)) return false;
  // チップの入力欄を編集中のときは、その中に新しいチップを作ったり装飾したりしない
  if (document.activeElement instanceof HTMLInputElement) return false;
  fn(active);
  return true;
}

/** フォーカス中のリッチテキスト部品のカーソル位置に文中数式を挿入する。できなければ false。 */
function insertInlineMath(): boolean {
  return withActiveRich((rich) => rich.insertChip('math'));
}

/** フォーカス中のリッチテキスト部品のカーソル位置に脚注を挿入する。できなければ false。 */
function insertFootnote(): boolean {
  return withActiveRich((rich) => rich.insertChip('footnote'));
}

/** 選択範囲の太字・斜体・下線を切り替える。できなければ false。 */
function applyInlineFormat(kind: FormatKind): boolean {
  return withActiveRich((rich) => rich.format(kind));
}

/** 選択範囲をリンクにする（カーソルがリンク上にあればURLの変更・解除）。できなければ false。 */
function editInlineLink(): boolean {
  return withActiveRich((rich) => rich.editLink());
}

/** カーソル位置の装飾の状態。パレットのボタンの押下表示に使う。リッチテキストの外なら null */
function inlineFormatState(): Record<FormatKind, boolean> | null {
  const active = activeRich;
  if (!active || !active.el.contains(document.activeElement) || document.activeElement instanceof HTMLInputElement) {
    return null;
  }
  return {
    bold: document.queryCommandState('bold'),
    italic: document.queryCommandState('italic'),
    underline: document.queryCommandState('underline'),
  };
}

interface ChipSpec {
  className: string;
  placeholder: string;
  render: (view: HTMLElement, value: string) => void;
}

const CHIP_SPECS: Record<ChipKind, ChipSpec> = {
  math: {
    className: 'inline-math',
    placeholder: 'TeX 例: x^2',
    render: (view, tex) => {
      if (tex.trim() === '') {
        view.textContent = '∑';
      } else if (typeof katex !== 'undefined') {
        katex.render(tex, view, { displayMode: false, throwOnError: false });
      } else {
        view.textContent = `$${tex}$`;
      }
    },
  },
  footnote: {
    className: 'inline-footnote',
    placeholder: '脚注の文章',
    render: (view, text) => {
      const mark = document.createElement('span');
      mark.className = 'inline-footnote-mark';
      mark.textContent = '注';
      const body = document.createElement('span');
      body.className = 'inline-footnote-text';
      body.textContent = text;
      view.append(mark, body);
      view.title = `脚注: ${text}`;
    },
  },
};

/** 装飾付きの要素から、その中の文字に効く装飾を求める。ブラウザが style 属性で装飾することもあるので両方見る */
function marksFor(node: HTMLElement, inherited: TextMarks): TextMarks {
  const marks: TextMarks = { ...inherited };
  const tag = node.tagName;
  if (tag === 'B' || tag === 'STRONG') marks.bold = true;
  if (tag === 'I' || tag === 'EM') marks.italic = true;
  if (tag === 'U') marks.underline = true;
  if (tag === 'A') {
    const href = node.getAttribute('href');
    if (href) marks.href = href;
  }
  const { fontWeight, fontStyle, textDecorationLine } = node.style;
  if (fontWeight) {
    if (fontWeight === 'bold' || fontWeight === 'bolder' || Number(fontWeight) >= 600) marks.bold = true;
    else delete marks.bold;
  }
  if (fontStyle) {
    if (fontStyle === 'italic' || fontStyle === 'oblique') marks.italic = true;
    else delete marks.italic;
  }
  if (textDecorationLine.includes('underline')) marks.underline = true;
  return marks;
}

function createRichTextEditor(options: RichTextOptions): HTMLElement {
  const el = document.createElement('div');
  el.className = 'block-input rich';
  el.contentEditable = 'true';
  el.dataset.placeholder = options.placeholder;
  el.spellcheck = false;

  const isChip = (node: Node | null): node is HTMLElement =>
    node instanceof HTMLElement && node.classList.contains('inline-chip');

  const serialize = (): RichText => {
    const out: RichText = [];
    const walk = (parent: Node, marks: TextMarks) => {
      for (const child of Array.from(parent.childNodes)) {
        if (child.nodeType === Node.TEXT_NODE) {
          out.push({ type: 'text', text: (child.textContent ?? '').replace(/​/g, ''), ...marks });
        } else if (isChip(child)) {
          const value = child.dataset.value ?? '';
          out.push(child.dataset.kind === 'footnote' ? { type: 'footnote', text: value } : { type: 'math', tex: value });
        } else if (child instanceof HTMLBRElement) {
          out.push({ type: 'text', text: '\n', ...marks });
        } else if (child instanceof HTMLElement) {
          walk(child, marksFor(child, marks)); // 太字などの要素は装飾を足して中身を拾う
        }
      }
    };
    walk(el, {});

    // 末尾の <br> はブラウザが付けるプレースホルダなので、改行1つ分を無視する（太字などの中にあることもある）
    let tail: Node | null = el.lastChild;
    while (
      tail instanceof HTMLElement &&
      !tail.classList.contains('inline-chip') &&
      !(tail instanceof HTMLBRElement) &&
      tail.lastChild
    ) {
      tail = tail.lastChild;
    }
    const content = normalizeRich(out);
    const last = content[content.length - 1];
    if (tail instanceof HTMLBRElement && last?.type === 'text' && last.text.endsWith('\n')) {
      last.text = last.text.slice(0, -1);
      if (last.text === '') content.pop();
    }
    return content;
  };

  const emit = () => {
    const content = serialize();
    // 全部消したあとに残る <br> や ZWSP を片付けて、placeholder(:empty) が効くようにする
    if (content.length === 0 && el.childNodes.length > 0) el.replaceChildren();
    el.querySelectorAll('a').forEach((a) => (a.title = a.getAttribute('href') ?? ''));
    options.onChange(content);
  };

  const placeCaret = (range: Range) => {
    const sel = window.getSelection();
    sel?.removeAllRanges();
    sel?.addRange(range);
  };

  /** チップの前後にカーソルを置ける場所（テキストノード）を用意する */
  const padChip = (chip: HTMLElement) => {
    if (!(chip.previousSibling instanceof Text)) chip.before(document.createTextNode(ZWSP));
    if (!(chip.nextSibling instanceof Text)) chip.after(document.createTextNode(ZWSP));
  };

  const caretAfter = (node: Node) => {
    el.focus();
    const range = document.createRange();
    range.setStartAfter(node);
    range.collapse(true);
    placeCaret(range);
  };

  const renderChip = (chip: HTMLElement) => {
    const view = document.createElement('span');
    view.className = 'inline-chip-view';
    CHIP_SPECS[chip.dataset.kind as ChipKind].render(view, chip.dataset.value ?? '');
    chip.replaceChildren(view);
  };

  const removeChip = (chip: HTMLElement) => {
    el.focus();
    const range = document.createRange();
    range.setStartBefore(chip);
    range.collapse(true);
    chip.remove();
    placeCaret(range);
    emit();
  };

  /** チップをその場で入力欄に切り替える。isNew は挿入直後（Esc で取り消すとチップごと消す）。 */
  const editChip = (chip: HTMLElement, isNew: boolean) => {
    const original = chip.dataset.value ?? '';
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'inline-chip-input';
    input.placeholder = CHIP_SPECS[chip.dataset.kind as ChipKind].placeholder;
    input.value = original;
    input.size = Math.max(8, original.length + 2);
    chip.replaceChildren(input);
    input.focus();

    let done = false;
    const finish = (how: 'commit' | 'cancel', refocus: boolean) => {
      if (done) return;
      done = true;
      if (how === 'cancel') chip.dataset.value = original;
      const empty = (chip.dataset.value ?? '').trim() === '';
      if (empty || (how === 'cancel' && isNew)) {
        if (refocus) removeChip(chip);
        else {
          chip.remove();
          emit();
        }
        return;
      }
      renderChip(chip);
      emit();
      if (refocus) caretAfter(chip);
    };

    input.addEventListener('input', () => {
      chip.dataset.value = input.value;
      input.size = Math.max(8, input.value.length + 2);
    });
    input.addEventListener('keydown', (e) => {
      e.stopPropagation(); // 外側の contenteditable の Enter / Backspace 処理に渡さない
      if (e.isComposing) return;
      if (e.key === 'Enter') {
        e.preventDefault();
        finish('commit', true);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        finish('cancel', true);
      } else if (e.key === 'Backspace' && input.value === '') {
        e.preventDefault();
        chip.dataset.value = '';
        finish('commit', true);
      }
    });
    input.addEventListener('blur', () => finish('commit', false));
  };

  const makeChip = (kind: ChipKind, value: string): HTMLElement => {
    const chip = document.createElement('span');
    chip.className = `inline-chip ${CHIP_SPECS[kind].className}`;
    chip.contentEditable = 'false';
    chip.dataset.kind = kind;
    chip.dataset.value = value;
    renderChip(chip);
    chip.addEventListener('click', () => {
      if (!chip.querySelector('input')) editChip(chip, false);
    });
    return chip;
  };

  /** 今のカーソル位置。部品の外にあれば末尾 */
  const currentRange = (): Range => {
    const sel = window.getSelection();
    if (sel && sel.rangeCount > 0 && el.contains(sel.anchorNode)) return sel.getRangeAt(0);
    const range = document.createRange();
    range.selectNodeContents(el);
    range.collapse(false);
    return range;
  };

  const insertChip = (kind: ChipKind) => {
    const range = currentRange();
    range.deleteContents();
    const chip = makeChip(kind, '');
    range.insertNode(chip);
    padChip(chip);
    editChip(chip, true);
  };

  const format = (kind: FormatKind) => {
    el.focus();
    // false にすると <span style="..."> ではなく <b> <i> <u> で装飾される
    document.execCommand('styleWithCSS', false, 'false');
    document.execCommand(kind);
    emit();
  };

  const editLink = () => {
    const range = currentRange().cloneRange();
    const start = range.startContainer;
    const anchor = (start instanceof Element ? start : start.parentElement)?.closest('a');
    const existing = anchor && el.contains(anchor) ? anchor : null;

    const answer = window.prompt(
      existing ? 'リンク先のURL（空にするとリンクを解除）' : 'リンク先のURL',
      existing?.getAttribute('href') ?? 'https://',
    );
    el.focus();
    placeCaret(range);
    if (answer === null) return;

    if (existing) {
      const url = normalizeLinkUrl(answer);
      if (answer.trim() === '') existing.replaceWith(...Array.from(existing.childNodes));
      else if (url) existing.setAttribute('href', url);
      else alert('http(s):// か mailto: で始まるURLを入力してください');
      emit();
      return;
    }

    const url = normalizeLinkUrl(answer);
    if (!url) {
      if (answer.trim() !== '' && answer.trim() !== 'https://') {
        alert('http(s):// か mailto: で始まるURLを入力してください');
      }
      return;
    }
    if (range.collapsed) {
      // 文字を選んでいなければ、URLそのものをリンクとして挿入する
      const a = document.createElement('a');
      a.href = url;
      a.textContent = url.replace(/^mailto:/i, '');
      range.insertNode(a);
      caretAfter(a);
    } else {
      document.execCommand('createLink', false, url);
    }
    emit();
  };

  /** モデルの text セグメント1つを、装飾の要素で包んだDOMにする */
  const textNode = (seg: TextSegment): Node => {
    const frag = document.createDocumentFragment();
    seg.text.split('\n').forEach((line, i) => {
      if (i > 0) frag.append(document.createElement('br'));
      if (line !== '') frag.append(document.createTextNode(line));
    });
    let node: Node = frag;
    const wrap = (tag: string) => {
      const wrapper = document.createElement(tag);
      wrapper.append(node);
      node = wrapper;
      return wrapper;
    };
    if (seg.underline) wrap('u');
    if (seg.italic) wrap('i');
    if (seg.bold) wrap('b');
    if (seg.href) wrap('a').setAttribute('href', seg.href);
    return node;
  };

  // モデル → DOM
  for (const seg of options.content) {
    if (seg.type === 'math') el.append(makeChip('math', seg.tex));
    else if (seg.type === 'footnote') el.append(makeChip('footnote', seg.text));
    else el.append(textNode(seg));
  }
  el.querySelectorAll<HTMLElement>('.inline-chip').forEach(padChip);
  el.querySelectorAll('a').forEach((a) => (a.title = a.getAttribute('href') ?? ''));

  el.addEventListener('input', emit);
  el.addEventListener('focusin', () => {
    activeRich = { el, insertChip, format, editLink };
  });

  el.addEventListener('keydown', (e) => {
    if (e.isComposing) return;
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
      const key = e.key.toLowerCase();
      const kind = ({ b: 'bold', i: 'italic', u: 'underline' } as const)[key as 'b' | 'i' | 'u'];
      if (kind || key === 'k') {
        // Ctrl+B は文章の外ではパレットの開閉なので、ここで止めて document まで届かせない
        e.preventDefault();
        e.stopPropagation();
        if (kind) format(kind);
        else editLink();
        return;
      }
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (!options.singleLine) document.execCommand('insertLineBreak');
      return;
    }
    // チップの直後（ZWSP の位置）での Backspace は、ZWSP ではなくチップを消す
    if (e.key === 'Backspace') {
      const sel = window.getSelection();
      const node = sel?.anchorNode;
      if (!sel || !sel.isCollapsed || !node) return;
      let prev: Node | null = null;
      if (node instanceof Text) {
        const atStart = sel.anchorOffset === 0 || (node.data.startsWith(ZWSP) && sel.anchorOffset === 1);
        if (atStart) prev = node.previousSibling;
      } else if (node === el && sel.anchorOffset > 0) {
        prev = el.childNodes[sel.anchorOffset - 1];
      }
      if (isChip(prev)) {
        e.preventDefault();
        removeChip(prev);
      }
    }
  });

  el.addEventListener('paste', (e) => {
    e.preventDefault();
    const raw = e.clipboardData?.getData('text/plain') ?? '';
    const text = options.singleLine ? raw.replace(/\r?\n/g, ' ') : raw.replace(/\r\n?/g, '\n');
    text.split('\n').forEach((line, i) => {
      if (i > 0) document.execCommand('insertLineBreak');
      if (line !== '') document.execCommand('insertText', false, line);
    });
  });

  return el;
}
