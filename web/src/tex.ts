/** 未指定の設定項目はこの値で出力する。marginMm は未指定なら geometry を読み込まない */
const DEFAULT_SETTINGS: Required<Omit<DocumentSettings, 'marginMm'>> = {
  paper: 'a4',
  landscape: false,
  fontSize: 11,
  twoColumn: false,
};

interface PaperSpec {
  label: string;
  /** jsarticle のクラスオプション */
  classOption: string;
  /** geometry に渡す用紙オプション。jsarticle の b5paper は JIS B5 だが、geometry の b5paper は ISO B5 なので名前が違う */
  geometryOption: string;
}

const PAPER_SIZES: Record<PaperSize, PaperSpec> = {
  a4: { label: 'A4', classOption: 'a4paper', geometryOption: 'a4paper' },
  a5: { label: 'A5', classOption: 'a5paper', geometryOption: 'a5paper' },
  b5: { label: 'B5', classOption: 'b5paper', geometryOption: 'b5j' },
  letter: { label: 'レター', classOption: 'letterpaper', geometryOption: 'letterpaper' },
};

const FONT_SIZES: FontSize[] = [10, 11, 12, 14];

/** 余白（mm）として受け付ける範囲。範囲外の値は丸めて出力する */
const MARGIN_MM_MIN = 5;
const MARGIN_MM_MAX = 50;

/**
 * uplatexオプションは必須。TeX Live 2022のjsarticleはこれが無いとupLaTeXでエラーになる。
 * dvipdfmx はクラスオプションに書くと graphicx / hyperref / geometry のすべてに伝わる。
 */
function documentClass(settings: DocumentSettings): string {
  const s = { ...DEFAULT_SETTINGS, ...settings };
  const options = [
    'uplatex',
    'dvipdfmx',
    PAPER_SIZES[s.paper].classOption,
    ...(s.landscape ? ['landscape'] : []),
    `${s.fontSize}pt`,
    ...(s.twoColumn ? ['twocolumn'] : []),
  ];
  return `\\documentclass[${options.join(',')}]{jsarticle}`;
}

/**
 * 余白の指定。jsarticle は 10pt 以外では \mag で文書全体を拡大するため、
 * truedimen と truemm を付けないと余白も用紙サイズも拡大される（11pt で A4 の幅が 230mm になる）。
 * 縦横の向きはクラスオプションの landscape が geometry にも伝わる。
 */
function geometryPackage(settings: DocumentSettings): string[] {
  if (settings.marginMm === undefined) return [];
  const paper = PAPER_SIZES[settings.paper ?? DEFAULT_SETTINGS.paper].geometryOption;
  const margin = Math.round(Math.min(MARGIN_MM_MAX, Math.max(MARGIN_MM_MIN, settings.marginMm)));
  return [`\\usepackage[truedimen,${paper},margin=${margin}truemm]{geometry}`];
}

const PACKAGES_BY_BLOCK_TYPE: Record<BlockType, string[]> = {
  title: [],
  heading: [],
  paragraph: [],
  list: [],
  math: ['\\usepackage{amsmath}'],
  toc: [],
  table: ['\\usepackage{booktabs}'],
  image: ['\\usepackage{graphicx}'],
  quote: [],
  // minted は shell-escape が必要で、コンパイルサーバーは -no-shell-escape で動かすため listings を使う
  code: [
    '\\usepackage{listings}',
    '\\lstset{basicstyle=\\ttfamily\\small,frame=single,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false,tabsize=4}',
  ],
  break: [],
};

/** 下線用。日本語の改行の都合で ulem の \uline を1文字ずつ使う（underlineTex を参照） */
const UNDERLINE_PACKAGES = ['\\usepackage[normalem]{ulem}'];

/** リンク用。hyperref は他のパッケージより後に読み込む。pxjahyper はしおりの日本語を文字化けさせないため */
const LINK_PACKAGES = [
  '\\usepackage{hyperref}',
  '\\usepackage{pxjahyper}',
  '\\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}',
];

const TEX_ESCAPES: Record<string, string> = {
  '\\': '\\textbackslash{}',
  '%': '\\%',
  '&': '\\&',
  '_': '\\_',
  '#': '\\#',
  '{': '\\{',
  '}': '\\}',
  '~': '\\textasciitilde{}',
  '^': '\\textasciicircum{}',
  '$': '\\$',
};

/** [平文] フィールド用。1文字ずつ置換するので二重エスケープにならない。 */
function escapeTex(text: string): string {
  return text.replace(/[\\%&_#{}~^$]/g, (c) => TEX_ESCAPES[c]);
}

/** この文字の前では改行させない（行頭禁則） */
const NO_BREAK_BEFORE = '、。，．・：；？！ー）」』】〕｝〉》';
/** この文字の後では改行させない（行末禁則） */
const NO_BREAK_AFTER = '（「『【〔｛〈《';

/**
 * 下線付きの文字列。ulem の \uline は空白でしか改行できないので、日本語をまとめて渡すと
 * 1行に収まらず本文からはみ出す。英数字の並び（空白を含む）はまとめて、それ以外は1文字ずつ \uline にし、
 * 間に \allowbreak を置いて改行できるようにする。隣り合う \uline の下線はつながって見える。
 */
function underlineTex(text: string): string {
  const tokens = text.match(/[\x20-\x7e]+|[^\x20-\x7e]/gu) ?? [];
  return tokens
    .map((token, i) => {
      const prev = tokens[i - 1];
      const canBreak =
        prev !== undefined && !NO_BREAK_BEFORE.includes(token[0]) && !NO_BREAK_AFTER.includes(prev[prev.length - 1]);
      return `${canBreak ? '\\allowbreak' : ''}\\uline{${escapeTex(token)}}`;
    })
    .join('');
}

/**
 * [その他] リンク先を \href の第1引数に使える形にする。使えないURLなら undefined（リンクにしない）。
 * \href は % と # を \% \# と書けば受け付ける。\ { } や空白などはTeXの区切りと衝突するので %XX にする。
 */
function hrefTex(href: string | undefined): string | undefined {
  const url = href?.trim();
  if (!url || !/^(https?:\/\/|mailto:)/i.test(url)) return undefined;
  return url
    .replace(/[\\{}^`"<>|\s]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase().padStart(2, '0')}`)
    .replace(/[%#]/g, (c) => `\\${c}`);
}

/** 1行分の平文に装飾を付ける。\href は他のコマンドの引数の中だと # や % を扱えないので一番外側に置く */
function markedTextTex(text: string, marks: TextMarks): string {
  let tex = marks.underline ? underlineTex(text) : escapeTex(text);
  if (marks.italic) tex = `\\textit{${tex}}`;
  if (marks.bold) tex = `\\textbf{${tex}}`;
  const href = hrefTex(marks.href);
  if (href) tex = `\\href{${href}}{${tex}}`;
  return tex;
}

const LINE_BREAK = Symbol('line break');

/**
 * 平文は escapeTex、文中数式は $...$、脚注は \footnote{} にして連結する。空の数式・脚注は出さない。
 * 改行は \newline にする。\\ だと次の行が [ で始まったときに行間の指定と読まれて壊れるため。
 * 段落の先頭・末尾の改行は捨て（行が無いところで改行するとエラーになる）、
 * 2つ目以降の連続した改行は空の行（\mbox{}）を挟んで空行にする。
 */
function richToTex(content: RichText): string {
  const pieces: (string | typeof LINE_BREAK)[] = [];
  for (const seg of content) {
    if (seg.type === 'math') {
      const tex = seg.tex.trim();
      if (tex !== '') pieces.push(`$${tex}$`);
    } else if (seg.type === 'footnote') {
      if (seg.text.trim() !== '') pieces.push(`\\footnote{${escapeTex(seg.text)}}`);
    } else {
      seg.text.split('\n').forEach((line, i) => {
        if (i > 0) pieces.push(LINE_BREAK);
        if (line !== '') pieces.push(markedTextTex(line, seg));
      });
    }
  }
  while (pieces[0] === LINE_BREAK) pieces.shift();
  while (pieces[pieces.length - 1] === LINE_BREAK) pieces.pop();
  return pieces
    .map((piece, i) => {
      if (piece !== LINE_BREAK) return piece;
      return pieces[i - 1] === LINE_BREAK ? '\\mbox{}\\newline\n' : '\\newline\n';
    })
    .join('');
}

interface ListMarkerSpec {
  /** 選択肢に出す名前 */
  label: string;
  /** エディタ上のマーカー表示。n は1始まりの項目番号 */
  glyph: (n: number) => string;
  env: 'itemize' | 'enumerate';
  /** 環境の中に置く \renewcommand。環境内なのでそのリストにしか効かない */
  setup?: string;
}

const DEFAULT_LIST_MARKER: ListMarker = 'dot';

const LIST_MARKERS: Record<ListMarker, ListMarkerSpec> = {
  dot: { label: '・', glyph: () => '・', env: 'itemize', setup: '\\renewcommand{\\labelitemi}{・}' },
  bullet: { label: '•', glyph: () => '•', env: 'itemize' },
  dash: { label: '–', glyph: () => '–', env: 'itemize', setup: '\\renewcommand{\\labelitemi}{--}' },
  number: { label: '1. 2. 3.', glyph: (n) => `${n}.`, env: 'enumerate' },
  paren: {
    label: '(1) (2) (3)',
    glyph: (n) => `(${n})`,
    env: 'enumerate',
    setup: '\\renewcommand{\\labelenumi}{(\\arabic{enumi})}',
  },
};

interface CodeLanguageSpec {
  label: string;
  /** listings の language= に渡す名前。未指定なら色分けしない */
  lst?: string;
}

const DEFAULT_CODE_LANGUAGE: CodeLanguage = 'plain';

const CODE_LANGUAGES: Record<CodeLanguage, CodeLanguageSpec> = {
  plain: { label: '言語なし' },
  python: { label: 'Python', lst: 'Python' },
  c: { label: 'C', lst: 'C' },
  cpp: { label: 'C++', lst: 'C++' },
  java: { label: 'Java', lst: 'Java' },
  ruby: { label: 'Ruby', lst: 'Ruby' },
  bash: { label: 'Bash', lst: 'bash' },
  sql: { label: 'SQL', lst: 'SQL' },
  html: { label: 'HTML', lst: 'HTML' },
  matlab: { label: 'MATLAB', lst: 'Matlab' },
};

/** コード中にあると lstlisting 環境が閉じてしまう文字列 */
const LSTLISTING_END = '\\end{lstlisting}';

/**
 * 和文などASCII以外の文字の並び。listings は upLaTeX の和文文字を正しい位置に出せない
 * （言語なしだと print("こんにちは") が printこんにちは ("") になる）ので、エスケープしてTeXの通常の組版に任せる。
 */
const NON_ASCII_RUN = /[^\x00-\x7f]+/g;
/** listings のエスケープ文字の候補。コード中に現れないものを1つ選ぶ */
const LST_ESCAPE_CANDIDATES = ['`', '|', '@', '!', '?', ';', ':', '+', '*', '<', '>'];

interface BreakSpec {
  label: string;
  tex: string;
}

const BREAK_KINDS: Record<BreakKind, BreakSpec> = {
  // \newpage は2段組みだと次の段へ進むだけなので、確実に次のページへ進む \clearpage を使う
  page: { label: '改ページ', tex: '\\clearpage' },
  small: { label: '空白（半行）', tex: '\\vspace{0.5\\baselineskip}' },
  medium: { label: '空白（1行）', tex: '\\vspace{\\baselineskip}' },
  large: { label: '空白（2行）', tex: '\\vspace{2\\baselineskip}' },
};

const HEADING_COMMANDS: Record<HeadingLevel, string> = {
  1: 'section',
  2: 'subsection',
  3: 'subsubsection',
};

/** ブロック1つ分のTeX */
function blockToTex(block: Block): string {
  switch (block.type) {
    case 'title':
      return '\\maketitle';
    case 'heading':
      return `\\${HEADING_COMMANDS[block.level]}{${escapeTex(block.text)}}`;
    case 'paragraph':
      return richToTex(block.content);
    case 'list': {
      const spec = LIST_MARKERS[block.marker ?? DEFAULT_LIST_MARKER];
      const items = block.items.map((item) => `  \\item ${richToTex(item.content)}`);
      const setup = spec.setup ? [`  ${spec.setup}`] : [];
      return [`\\begin{${spec.env}}`, ...setup, ...items, `\\end{${spec.env}}`].join('\n');
    }
    case 'math': {
      if (!block.displayMode) return `$${block.tex}$`;
      const env = block.numbered ? 'align' : 'align*';
      return `\\begin{${env}}\n${block.tex}\n\\end{${env}}`;
    }
    case 'toc':
      return '\\tableofcontents';
    case 'table':
      return tableToTex(block);
    case 'image':
      return imageToTex(block);
    case 'quote':
      return `\\begin{quote}\n${richToTex(block.content)}\n\\end{quote}`;
    case 'code':
      return codeToTex(block);
    case 'break':
      return BREAK_KINDS[block.kind].tex;
  }
}

function codeToTex(block: CodeBlock): string {
  if (block.code.trim() === '') return '% コードが未入力';
  const lst = CODE_LANGUAGES[block.language].lst;
  const options = lst ? [`language=${lst}`] : [];
  // \end と { の間に空白を入れて崩す。PDF上では空白が1つ増えて見える
  let code = block.code.split(LSTLISTING_END).join('\\end {lstlisting}');
  // 候補を全部使っているコードはまず無いが、その場合はエスケープせずに出す（日本語の位置がずれるだけ）
  const escape = LST_ESCAPE_CANDIDATES.find((c) => !code.includes(c));
  if (escape && /[^\x00-\x7f]/.test(code)) {
    options.push(`escapechar=${escape}`);
    code = code.replace(NON_ASCII_RUN, (run) => `${escape}${run}${escape}`);
  }
  const optionText = options.length ? `[${options.join(',')}]` : '';
  return `\\begin{lstlisting}${optionText}\n${code}\n\\end{lstlisting}`;
}

function tableToTex(block: TableBlock): string {
  const row = (r: TableRow) => `    ${r.cells.map(escapeTex).join(' & ')} \\\\`;
  const lines = [
    '\\begin{table}[htbp]',
    '  \\centering',
    ...(block.caption ? [`  \\caption{${escapeTex(block.caption)}}`] : []),
    `  \\begin{tabular}{${block.columns.map((c) => c.align).join('')}}`,
    '    \\toprule',
    ...(block.header ? [row(block.header), '    \\midrule'] : []),
    ...block.rows.map(row),
    '    \\bottomrule',
    '  \\end{tabular}',
    '\\end{table}',
  ];
  return lines.join('\n');
}

/** 小数3桁に丸めて、余計な 0 を落とす */
function ratio(n: number): string {
  return String(Number(n.toFixed(3)));
}

function imageToTex(block: ImageBlock): string {
  if (!block.src) return '% 画像が未選択';
  const width = block.widthRatio ?? 1;
  const options = block.widthRatio === undefined ? '' : `[width=${ratio(width)}\\linewidth]`;
  const graphic = `\\includegraphics${options}{${block.src}}`;
  const caption = block.caption ? [`  \\caption{${escapeTex(block.caption)}}`] : [];

  if (block.offsetX === undefined && block.offsetY === undefined) {
    return ['\\begin{figure}[htbp]', '  \\centering', `  ${graphic}`, ...caption, '\\end{figure}'].join('\n');
  }

  // 配置モード: 本文の流れに置いたまま、縦ずれ・横ずれで位置を再現する。
  // \vspace* は縦モードで先に置く（横モードだと次の行に効く）。\centering は横ずれごと寄ってしまうので使わない。
  const x = block.offsetX ?? (1 - width) / 2;
  const y = block.offsetY ?? 0;
  return [
    '\\begin{figure}[htbp]',
    ...(y > 0 ? [`  \\vspace*{${ratio(y)}\\linewidth}`] : []),
    `  \\noindent\\hspace*{${ratio(x)}\\linewidth}${graphic}`,
    ...(caption.length ? ['  \\par', ...caption] : []),
    '\\end{figure}',
  ].join('\n');
}

/** ブロックが持つリッチテキストをすべて返す（文中の装飾に必要なパッケージを調べるため） */
function richContents(block: Block): RichText[] {
  switch (block.type) {
    case 'paragraph':
    case 'quote':
      return [block.content];
    case 'list':
      return block.items.map((item) => item.content);
    default:
      return [];
  }
}

function documentToTex(doc: BlockDocument): string {
  const settings = doc.settings ?? {};
  const packages: string[] = [...geometryPackage(settings)];
  const addPackages = (list: string[]) => {
    for (const pkg of list) {
      if (!packages.includes(pkg)) packages.push(pkg);
    }
  };
  for (const block of doc.blocks) addPackages(PACKAGES_BY_BLOCK_TYPE[block.type]);

  const segments = doc.blocks.flatMap(richContents).flat();
  const texts = segments.filter((seg): seg is Extract<InlineSegment, { type: 'text' }> => seg.type === 'text');
  if (texts.some((seg) => seg.underline)) addPackages(UNDERLINE_PACKAGES);
  if (texts.some((seg) => hrefTex(seg.href) !== undefined)) addPackages(LINK_PACKAGES);

  const titleBlock = doc.blocks.find((b): b is TitleBlock => b.type === 'title');
  const titleLines = titleBlock
    ? [
        `\\title{${escapeTex(titleBlock.text)}}`,
        `\\author{${escapeTex(titleBlock.author)}}`,
        `\\date{${escapeTex(titleBlock.date)}}`,
      ]
    : [];
  const head = [documentClass(settings), ...packages, ...titleLines].join('\n');
  const body = doc.blocks.map(blockToTex).join('\n\n');
  return `${head}\n\n\\begin{document}\n\n${body}${body ? '\n\n' : ''}\\end{document}\n`;
}
