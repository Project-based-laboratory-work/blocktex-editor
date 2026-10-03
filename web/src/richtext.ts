// KaTeX は CDN から読む。オフラインなどで読めない場合は undefined になるので、使う前に typeof で確認する。
declare const katex:
  | { render(tex: string, el: HTMLElement, options: { displayMode: boolean; throwOnError: boolean }): void }
  | undefined;

/** カーソルをチップの前後に置けるよう、チップの両隣に入れておく幅ゼロの文字。保存時には取り除く。 */
const ZWSP = '​';

/** 隣り合う text を結合し、空の text を捨てる */
function normalizeRich(content: RichText): RichText {
  const out: RichText = [];
  for (const seg of content) {
    const last = out[out.length - 1];
    if (seg.type === 'text') {
      if (seg.text === '') continue;
      if (last?.type === 'text') last.text += seg.text;
      else out.push({ type: 'text', text: seg.text });
    } else {
      out.push({ type: 'math', tex: seg.tex });
    }
  }
  return out;
}

interface RichTextOptions {
  content: RichText;
  /** true: 改行を入れられない（箇条書きの項目用）。Enter の扱いは呼び出し側に任せる */
  singleLine: boolean;
  placeholder: string;
  onChange: (content: RichText) => void;
}

/** 直近でフォーカスしたリッチテキスト部品。パレットのボタンやショートカットから数式を挿入するために覚えておく。 */
let activeRichInsert: { el: HTMLElement; insertMath: () => void } | null = null;

/** フォーカス中のリッチテキスト部品のカーソル位置に文中数式を挿入する。できなければ false。 */
function insertInlineMath(): boolean {
  const active = activeRichInsert;
  if (!active || !active.el.isConnected || !active.el.contains(document.activeElement)) return false;
  // 数式の入力欄を編集中のときは、その中に新しいチップを作らない
  if (document.activeElement instanceof HTMLInputElement) return false;
  active.insertMath();
  return true;
}

function createRichTextEditor(options: RichTextOptions): HTMLElement {
  const el = document.createElement('div');
  el.className = 'block-input rich';
  el.contentEditable = 'true';
  el.dataset.placeholder = options.placeholder;
  el.spellcheck = false;

  const isChip = (node: Node | null): node is HTMLElement =>
    node instanceof HTMLElement && node.classList.contains('inline-math');

  const serialize = (): RichText => {
    const out: RichText = [];
    let text = '';
    const flush = () => {
      if (text !== '') out.push({ type: 'text', text });
      text = '';
    };
    const walk = (parent: Node) => {
      for (const child of Array.from(parent.childNodes)) {
        if (child.nodeType === Node.TEXT_NODE) {
          text += (child.textContent ?? '').replace(/\u200b/g, '');
        } else if (isChip(child)) {
          flush();
          out.push({ type: 'math', tex: child.dataset.tex ?? '' });
        } else if (child instanceof HTMLBRElement) {
          text += '\n';
        } else if (child instanceof HTMLElement) {
          walk(child); // ブラウザが挟んだ余計な要素は中身だけ拾う
        }
      }
    };
    walk(el);
    // 末尾の <br> はブラウザが付けるプレースホルダなので、改行1つ分を無視する
    if (el.lastChild instanceof HTMLBRElement && text.endsWith('\n')) text = text.slice(0, -1);
    flush();
    return normalizeRich(out);
  };

  const emit = () => {
    const content = serialize();
    // 全部消したあとに残る <br> や ZWSP を片付けて、placeholder(:empty) が効くようにする
    if (content.length === 0 && el.childNodes.length > 0) el.replaceChildren();
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
    const tex = chip.dataset.tex ?? '';
    const view = document.createElement('span');
    view.className = 'inline-math-view';
    if (tex.trim() === '') {
      view.textContent = '∑';
    } else if (typeof katex !== 'undefined') {
      katex.render(tex, view, { displayMode: false, throwOnError: false });
    } else {
      view.textContent = `$${tex}$`;
    }
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

  /** チップをその場でTeX入力欄に切り替える。isNew は挿入直後（Esc で取り消すとチップごと消す）。 */
  const editChip = (chip: HTMLElement, isNew: boolean) => {
    const original = chip.dataset.tex ?? '';
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'inline-math-input';
    input.placeholder = 'TeX 例: x^2';
    input.value = original;
    input.size = Math.max(8, original.length + 2);
    chip.replaceChildren(input);
    input.focus();

    let done = false;
    const finish = (how: 'commit' | 'cancel', refocus: boolean) => {
      if (done) return;
      done = true;
      if (how === 'cancel') chip.dataset.tex = original;
      const empty = (chip.dataset.tex ?? '').trim() === '';
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
      chip.dataset.tex = input.value;
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
        chip.dataset.tex = '';
        finish('commit', true);
      }
    });
    input.addEventListener('blur', () => finish('commit', false));
  };

  const makeChip = (tex: string): HTMLElement => {
    const chip = document.createElement('span');
    chip.className = 'inline-math';
    chip.contentEditable = 'false';
    chip.dataset.tex = tex;
    renderChip(chip);
    chip.addEventListener('click', () => {
      if (!chip.querySelector('input')) editChip(chip, false);
    });
    return chip;
  };

  const insertMath = () => {
    const sel = window.getSelection();
    let range: Range;
    if (sel && sel.rangeCount > 0 && el.contains(sel.anchorNode)) {
      range = sel.getRangeAt(0);
    } else {
      range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
    }
    range.deleteContents();
    const chip = makeChip('');
    range.insertNode(chip);
    padChip(chip);
    editChip(chip, true);
  };

  // モデル → DOM
  for (const seg of options.content) {
    if (seg.type === 'math') {
      el.append(makeChip(seg.tex));
      continue;
    }
    seg.text.split('\n').forEach((line, i) => {
      if (i > 0) el.append(document.createElement('br'));
      if (line !== '') el.append(document.createTextNode(line));
    });
  }
  el.querySelectorAll<HTMLElement>('.inline-math').forEach(padChip);

  el.addEventListener('input', emit);
  el.addEventListener('focusin', () => {
    activeRichInsert = { el, insertMath };
  });

  el.addEventListener('keydown', (e) => {
    if (e.isComposing) return;
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
