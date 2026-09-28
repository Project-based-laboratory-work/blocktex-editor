// 文字列フィールドの扱いは2種類ある。エスケープ処理はこの区別に従うこと。
//   [平文]  ユーザーが日本語を打ち込む欄。TeX出力時に % & _ # { } ~ ^ \ をエスケープする。
//   [生TeX] ユーザーがTeX記法を直接書く欄。エスケープせずそのまま出力する。
// 各フィールドのコメントで [平文] / [生TeX] / [その他] を明記している。

/**
 * @typedef {string} BlockId
 * @typedef {1 | 2 | 3} HeadingLevel section / subsection / subsubsection
 *
 * @typedef {object} HeadingBlock
 * @property {BlockId} id
 * @property {'heading'} type
 * @property {HeadingLevel} level
 * @property {string} text [平文] 見出し文字列
 *
 * @typedef {object} ParagraphBlock
 * @property {BlockId} id
 * @property {'paragraph'} type
 * @property {string} text [平文] 段落本文
 *
 * @typedef {object} ListBlock
 * @property {BlockId} id
 * @property {'list'} type
 * @property {string[]} items [平文] 各項目。itemize の \item 1つに対応する
 *
 * @typedef {object} MathBlock
 * @property {BlockId} id
 * @property {'math'} type
 * @property {boolean} displayMode true: 別行立て（align環境）。false: 文中扱い（$...$ を単独段落として出力）
 * @property {string} tex [生TeX] 数式本体。エスケープしない
 *
 * @typedef {'l' | 'c' | 'r'} TableCellAlign
 *
 * @typedef {object} TableColumn
 * @property {TableCellAlign} align
 *
 * @typedef {string} TableCell [平文] セル1つ分の文字列
 *
 * 表の1行。`string[]` ではなくオブジェクトで包んでいる。
 * Firestore は配列の要素に配列を置けない（nested array 非対応）ため、
 * rows を string[][] にすると Phase3 でブロック配列をそのまま保存できなくなるため。
 * 詳細は BACKEND.md「0-5-3 保存データのスキーマ設計」を参照。
 * @typedef {object} TableRow
 * @property {TableCell[]} cells
 *
 * @typedef {object} TableBlock
 * @property {BlockId} id
 * @property {'table'} type
 * @property {string} [caption] [平文] 表のキャプション。未指定なら \caption を出力しない
 * @property {TableColumn[]} columns 列定義。tabular の列指定（例: {lc}）はこの順序で組み立てる
 * @property {TableRow} [header] 見出し行。未指定なら \toprule の直後に \midrule を出さず、本体行のみを出力する
 * @property {TableRow[]} rows 本体行。各行の cells.length は columns.length と一致させる。
 *   型では保証できないため、編集UI側で列の追加削除時に揃えること。
 *
 * @typedef {object} ImageBlock
 * @property {BlockId} id
 * @property {'image'} type
 * @property {string} src [その他] 画像の参照先。ファイル名やURLであり、エスケープも生TeX展開もしない
 * @property {string} [caption] [平文] 図のキャプション。未指定なら \caption を出力しない
 * @property {number} [widthRatio] \linewidth に対する幅の比率（0〜1）。未指定なら等倍
 *
 * @typedef {HeadingBlock | ParagraphBlock | ListBlock | MathBlock | TableBlock | ImageBlock} Block
 * @typedef {Block['type']} BlockType
 *
 * 保存・読み込みの単位となるブロック文書。
 * schemaVersion はリテラル型ではなく number にしている。
 * 読み込むJSONは旧バージョンでもありうるため、型で「常に最新版」と決め打ちすると
 * 「読み込んだが未移行の文書」を表現できず、移行処理そのものが書けなくなるため。
 * 最新版かどうかは isCurrentSchema() で判定する。
 * @typedef {object} BlockDocument
 * @property {number} schemaVersion
 * @property {string} id
 * @property {string} title
 * @property {Block[]} blocks
 *
 * Firestore の users/{uid}/projects/{projectId} 1件に対応する（Phase3）。
 * BlockDocument との対応:
 *   CloudProject.id   ← FirestoreのドキュメントID。ドキュメント本体のフィールドには持たせない
 *   CloudProject.name ← BlockDocument.title と同じもの
 *   blocks            ← BlockDocument.blocks をそのまま（ローカル保存と同じ形式）
 * Firestore の Timestamp ↔ Date の変換はPhase3でSDKとの境界で行う。
 * @typedef {object} CloudProject
 * @property {string} id
 * @property {string} name
 * @property {number} schemaVersion
 * @property {Block[]} blocks
 * @property {Date} createdAt
 * @property {Date} updatedAt
 */

/** 現行のスキーマ版数。ブロックの形を変えたらインクリメントし、移行処理を書く。 */
const SCHEMA_VERSION = 1;

/**
 * 読み込んだ文書が現行スキーマかどうか。false なら移行処理が必要。
 * @param {BlockDocument} doc
 * @returns {boolean}
 */
function isCurrentSchema(doc) {
  return doc.schemaVersion === SCHEMA_VERSION;
}

/**
 * ブロック・文書のID生成。
 *
 * crypto.randomUUID() は secure context（https または localhost）でしか定義されない。
 * LAN内アドレス（http://192.168.x.x など）でアクセスすると undefined になり、
 * ブロック追加時に落ちる。チームでの動作確認や発表デモで踏みうるため、フォールバックを用意する。
 * @returns {string}
 */
function generateId() {
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
