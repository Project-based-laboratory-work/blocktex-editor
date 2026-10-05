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

/**
 * 文中に数式を混ぜられる文字列。段落本文や箇条書きの項目に使う。
 * 数式は $...$ の中身だけを持つ（$ 自体は TeX 出力時に付ける）。
 */
type InlineSegment =
  | { type: 'text'; text: string } // [平文] 改行は '\n'
  | { type: 'math'; tex: string }; // [生TeX]
type RichText = InlineSegment[];

interface ParagraphBlock {
  id: BlockId;
  type: 'paragraph';
  /** 段落本文（平文と文中数式の並び） */
  content: RichText;
}

/** 箇条書きのマーカー。表示とTeX出力は tex.ts の LIST_MARKERS で定義する */
type ListMarker = 'dot' | 'bullet' | 'dash' | 'number' | 'paren';

interface ListBlock {
  id: BlockId;
  type: 'list';
  /** [その他] マーカーの種類。未指定は 'dot'（・）として扱う。後から足した項目なので optional */
  marker?: ListMarker;
  /** 各項目。itemize の \item 1つに対応する */
  items: ListItem[];
}

/**
 * 箇条書きの1項目。`RichText[]` ではなくオブジェクトで包んでいる。
 * Firestore は入れ子配列を保存できないため（TableRow と同じ理由）。
 */
interface ListItem {
  content: RichText;
}

interface MathBlock {
  id: BlockId;
  type: 'math';
  /** true: 別行立て（align環境）。false: 文中扱い（$...$ を単独段落として出力） */
  displayMode: boolean;
  /** [生TeX] 数式本体。エスケープしない */
  tex: string;
}

/** 文書タイトル。1文書に1つだけで、常に先頭に置く（\maketitle に対応） */
interface TitleBlock {
  id: BlockId;
  type: 'title';
  /** [平文] タイトル */
  text: string;
  /** [平文] 著者。空なら \author{} */
  author: string;
  /** [平文] 日付。空なら \date{}（日付を出さない） */
  date: string;
}

interface TocBlock {
  id: BlockId;
  type: 'toc';
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
  /**
   * 画像の左端の位置（\linewidth に対する比率。0〜1-widthRatio）。
   * offsetX / offsetY がどちらも未指定なら中央寄せ。どちらかが指定されたら配置モードで、
   * 未指定側は X=中央、Y=0 として扱う。
   */
  offsetX?: number;
  /** ブロック先頭から画像上端までの余白（\linewidth に対する比率。0〜1） */
  offsetY?: number;
}

type Block =
  | TitleBlock
  | HeadingBlock
  | ParagraphBlock
  | ListBlock
  | MathBlock
  | TocBlock
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
  /** 文書の名前（ファイル名など）。TeXのタイトルは TitleBlock が持つ */
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

/**
 * 現行のスキーマ版数。ブロックの形を変えたらインクリメントし、移行処理を書く。
 * v2: 段落の text → content、箇条書きの items: string[] → ListItem[]（文中数式のため）。
 *     保存機能が未実装で旧形式のデータは存在しないため、v1 からの移行処理は書いていない。
 */
const SCHEMA_VERSION = 2;

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
