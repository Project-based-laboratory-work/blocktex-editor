// uplatexオプションは必須。TeX Live 2022のjsarticleはこれが無いとupLaTeXでエラーになる。
const DOCUMENT_CLASS = '\\documentclass[uplatex,a4paper,11pt]{jsarticle}';

const PACKAGES_BY_BLOCK_TYPE: Record<BlockType, string[]> = {
  heading: [],
  paragraph: [],
  list: [],
  math: ['\\usepackage{amsmath}'],
  table: ['\\usepackage{booktabs}'],
  image: ['\\usepackage[dvipdfmx]{graphicx}'],
};

const BASE_PACKAGES: string[] = [];
