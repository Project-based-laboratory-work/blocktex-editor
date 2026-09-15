import type { BlockType } from '../model/block';

export const DOCUMENT_CLASS = '\\documentclass[a4paper,11pt]{jsarticle}';

export const PACKAGES_BY_BLOCK_TYPE: Record<BlockType, string[]> = {
  heading: [],
  paragraph: [],
  list: [],
  math: ['\\usepackage{amsmath}'],
  table: ['\\usepackage{booktabs}'],
  image: ['\\usepackage[dvipdfmx]{graphicx}'],
};

export const BASE_PACKAGES: string[] = [];
