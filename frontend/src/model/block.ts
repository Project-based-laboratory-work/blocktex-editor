// 文字列フィールドの扱いは2種類ある。Phase1のエスケープ処理（1-5-1）はこの区別に従うこと。
//   [平文]  ユーザーが日本語を打ち込む欄。TeX出力時に % & _ # { } ~ ^ \ をエスケープする。
//   [生TeX] ユーザーがTeX記法を直接書く欄。エスケープせずそのまま出力する。
// 各フィールドのコメントで [平文] / [生TeX] / [その他] を明記している。

export type BlockId = string;

export type HeadingLevel = 1 | 2 | 3; // section / subsection / subsubsection

export interface HeadingBlock {
  id: BlockId;
  type: 'heading';
  level: HeadingLevel;
  /** [平文] 見出し文字列 */
  text: string;
}

export interface ParagraphBlock {
  id: BlockId;
  type: 'paragraph';
  /** [平文] 段落本文 */
  text: string;
}

export interface ListBlock {
  id: BlockId;
  type: 'list';
  /** [平文] 各項目。itemize の \item 1つに対応する */
  items: string[];
}

export interface MathBlock {
  id: BlockId;
  type: 'math';
  /** true: 別行立て（align環境）。false: 文中扱い（$...$ を単独段落として出力） */
  displayMode: boolean;
  /** [生TeX] 数式本体。エスケープしない */
  tex: string;
}

export type TableCellAlign = 'l' | 'c' | 'r';

export interface TableColumn {
  align: TableCellAlign;
}

/** [平文] セル1つ分の文字列 */
export type TableCell = string;

/**
 * 表の1行。`string[]` ではなくオブジェクトで包んでいる。
 * Firestore は配列の要素に配列を置けない（nested array 非対応）ため、
 * rows を string[][] にすると Phase3 でブロック配列をそのまま保存できなくなるため。
 * 詳細は BACKEND.md「0-5-3 保存データのスキーマ設計」を参照。
 */
export interface TableRow {
  cells: TableCell[];
}

export interface TableBlock {
  id: BlockId;
  type: 'table';
  /** [平文] 表のキャプション。未指定なら \caption を出力しない */
  caption?: string;
  /** 列定義。tabular の列指定（例: {lc}）はこの順序で組み立てる */
  columns: TableColumn[];
  /** 見出し行。未指定なら \toprule の直後に \midrule を出さず、本体行のみを出力する */
  header?: TableRow;
  /**
   * 本体行。各行の cells.length は columns.length と一致させる。
   * 型では保証できないため、編集UI（Phase2 / 2-1-2）側で列の追加削除時に揃えること。
   */
  rows: TableRow[];
}

export interface ImageBlock {
  id: BlockId;
  type: 'image';
  /** [その他] 画像の参照先。ファイル名やURLであり、エスケープも生TeX展開もしない */
  src: string;
  /** [平文] 図のキャプション。未指定なら \caption を出力しない */
  caption?: string;
  /** \linewidth に対する幅の比率（0〜1）。未指定なら等倍 */
  widthRatio?: number;
}

export type Block = HeadingBlock | ParagraphBlock | ListBlock | MathBlock | TableBlock | ImageBlock;

export type BlockType = Block['type'];
