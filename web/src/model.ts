// 文字列フィールドの扱いは2種類ある。エスケープ処理はこの区別に従うこと。
//   [平文]  ユーザーが日本語を打ち込む欄。TeX出力時に % & _ # { } ~ ^ \ をエスケープする。
//   [生TeX] ユーザーがTeX記法を直接書く欄。エスケープせずそのまま出力する。
// 各フィールドのコメントで [平文] / [生TeX] / [その他] を明記している。

type BlockId = string;

/** section / subsection / subsubsection */
type HeadingLevel = 1 | 2 | 3;

interface HeadingBlock {
  id: BlockId;
  type: 'heading';
  level: HeadingLevel;
  /** [平文] 見出し文字列 */
  text: string;
}

interface ParagraphBlock {
  id: BlockId;
  type: 'paragraph';
  /** [平文] 段落本文 */
  text: string;
}

interface ListBlock {
  id: BlockId;
  type: 'list';
  /** [平文] 各項目。itemize の \item 1つに対応する */
  items: string[];
}

interface MathBlock {
  id: BlockId;
  type: 'math';
  /** true: 別行立て（align環境）。false: 文中扱い（$...$ を単独段落として出力） */
  displayMode: boolean;
  /** [生TeX] 数式本体。エスケープしない */
  tex: string;
}

type TableCellAlign = 'l' | 'c' | 'r';

interface TableColumn {
  align: TableCellAlign;
}

/** [平文] セル1つ分の文字列 */
type TableCell = string;

/**
 * 表の1行。`string[]` ではなくオブジェクトで包んでいる。
 * Firestore は配列の要素に配列を置けない（nested array 非対応）ため、
 * rows を string[][] にすると Phase3 でブロック配列をそのまま保存できなくなるため。
 * 詳細は BACKEND.md「0-5-3 保存データのスキーマ設計」を参照。
 */
interface TableRow {
  cells: TableCell[];
}

interface TableBlock {
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
   * 型では保証できないため、編集UI側で列の追加削除時に揃えること。
   */
  rows: TableRow[];
}

interface ImageBlock {
  id: BlockId;
  type: 'image';
  /** [その他] 画像の参照先。ファイル名やURLであり、エスケープも生TeX展開もしない */
  src: string;
  /** [平文] 図のキャプション。未指定なら \caption を出力しない */
  caption?: string;
  /** \linewidth に対する幅の比率（0〜1）。未指定なら等倍 */
  widthRatio?: number;
}

type Block =
  | HeadingBlock
  | ParagraphBlock
  | ListBlock
  | MathBlock
  | TableBlock
  | ImageBlock;

type BlockType = Block['type'];

/**
 * 保存・読み込みの単位となるブロック文書。
 * schemaVersion はリテラル型ではなく number にしている。
 * 読み込むJSONは旧バージョンでもありうるため、型で「常に最新版」と決め打ちすると
 * 「読み込んだが未移行の文書」を表現できず、移行処理そのものが書けなくなるため。
 * 最新版かどうかは isCurrentSchema() で判定する。
 */
interface BlockDocument {
  schemaVersion: number;
  id: string;
  title: string;
  blocks: Block[];
}

/**
 * Firestore の users/{uid}/projects/{projectId} 1件に対応する（Phase3）。
 * BlockDocument との対応:
 *   CloudProject.id   ← FirestoreのドキュメントID。ドキュメント本体のフィールドには持たせない
 *   CloudProject.name ← BlockDocument.title と同じもの
 *   blocks            ← BlockDocument.blocks をそのまま（ローカル保存と同じ形式）
 * Firestore の Timestamp ↔ Date の変換はPhase3でSDKとの境界で行う。
 */
interface CloudProject {
  id: string;
  name: string;
  schemaVersion: number;
  blocks: Block[];
  createdAt: Date;
  updatedAt: Date;
}

/** 現行のスキーマ版数。ブロックの形を変えたらインクリメントし、移行処理を書く。 */
const SCHEMA_VERSION = 1;

/** 読み込んだ文書が現行スキーマかどうか。false なら移行処理が必要。 */
function isCurrentSchema(doc: BlockDocument): boolean {
  return doc.schemaVersion === SCHEMA_VERSION;
}

/**
 * ブロック・文書のID生成。
 *
 * crypto.randomUUID() は secure context（https または localhost）でしか定義されない。
 * LAN内アドレス（http://192.168.x.x など）でアクセスすると undefined になり、
 * ブロック追加時に落ちる。チームでの動作確認や発表デモで踏みうるため、フォールバックを用意する。
 */
function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  // UUID v4 形式（衝突しなければよいだけなので暗号強度は要求しない）
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.floor(Math.random() * 16);
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
