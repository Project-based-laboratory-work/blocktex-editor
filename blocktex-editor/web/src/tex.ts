// uplatexオプションは必須。TeX Live 2022のjsarticleはこれが無いとupLaTeXでエラーになる。
const DOCUMENT_CLASS = '\\documentclass[uplatex,a4paper,11pt]{jsarticle}';

const PACKAGES_BY_BLOCK_TYPE: Record<BlockType, string[]> = {
  title: [],
  heading: [],
  paragraph: [],
  list: [],
  math: ['\\usepackage{amsmath}'],
  toc: [],
  table: ['\\usepackage{booktabs}'],
  image: ['\\usepackage[dvipdfmx]{graphicx}'],
};

const BASE_PACKAGES: string[] = [];

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

/** 平文は escapeTex、文中数式は $...$ にして連結する。空の数式は出さない。 */
function richToTex(content: RichText): string {
  return content
    .map((seg) => {
      if (seg.type === 'text') return escapeTex(seg.text);
      const tex = seg.tex.trim();
      return tex === '' ? '' : `$${tex}$`;
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

const HEADING_COMMANDS: Record<HeadingLevel, string> = {
  1: 'section',
  2: 'subsection',
  3: 'subsubsection',
};

/** ブロック1つ分のTeX。表・画像は未対応のため、コメント行を返す。 */
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
    case 'math':
      return block.displayMode
        ? `\\begin{align*}\n${block.tex}\n\\end{align*}`
        : `$${block.tex}$`;
    case 'toc':
      return '\\tableofcontents';
    case 'table':
      return tableToTex(block);
    case 'image':
      return imageToTex(block);
  }
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

function documentToTex(doc: BlockDocument): string {
  const packages: string[] = [...BASE_PACKAGES];
  for (const block of doc.blocks) {
    for (const pkg of PACKAGES_BY_BLOCK_TYPE[block.type]) {
      if (!packages.includes(pkg)) packages.push(pkg);
    }
  }

  const titleBlock = doc.blocks.find((b): b is TitleBlock => b.type === 'title');
  const titleLines = titleBlock
    ? [
        `\\title{${escapeTex(titleBlock.text)}}`,
        `\\author{${escapeTex(titleBlock.author)}}`,
        `\\date{${escapeTex(titleBlock.date)}}`,
      ]
    : [];
  const head = [DOCUMENT_CLASS, ...packages, ...titleLines].join('\n');
  const body = doc.blocks.map(blockToTex).join('\n\n');
  return `${head}\n\n\\begin{document}\n\n${body}${body ? '\n\n' : ''}\\end{document}\n`;
}
