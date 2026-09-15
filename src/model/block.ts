export type BlockId = string;

export type HeadingLevel = 1 | 2 | 3; // section / subsection / subsubsection

export interface HeadingBlock {
  id: BlockId;
  type: 'heading';
  level: HeadingLevel;
  text: string;
}

export interface ParagraphBlock {
  id: BlockId;
  type: 'paragraph';
  text: string;
}

export interface ListBlock {
  id: BlockId;
  type: 'list';
  items: string[];
}

export interface MathBlock {
  id: BlockId;
  type: 'math';
  displayMode: boolean;
  tex: string;
}

export type TableCellAlign = 'l' | 'c' | 'r';

export interface TableColumn {
  align: TableCellAlign;
}

export type TableCell = string;
export type TableRow = TableCell[];

export interface TableBlock {
  id: BlockId;
  type: 'table';
  caption?: string;
  columns: TableColumn[];
  header?: TableRow;
  rows: TableRow[];
}

export interface ImageBlock {
  id: BlockId;
  type: 'image';
  src: string;
  caption?: string;
  widthRatio?: number;
}

export type Block = HeadingBlock | ParagraphBlock | ListBlock | MathBlock | TableBlock | ImageBlock;

export type BlockType = Block['type'];
