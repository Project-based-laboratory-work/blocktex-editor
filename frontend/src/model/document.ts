import type { Block } from './block';

/** 現行のスキーマ版数。ブロックの形を変えたらインクリメントし、移行処理を書く。 */
export const SCHEMA_VERSION = 1;

/**
 * 保存・読み込みの単位となるブロック文書。
 *
 * schemaVersion はリテラル型（`1`）ではなく number にしている。
 * 読み込むJSONは旧バージョンでもありうるため、型で「常に最新版」と決め打ちすると
 * 「読み込んだが未移行の文書」を表現できず、移行処理そのものが書けなくなるため。
 * 最新版かどうかは isCurrentSchema() で判定する。
 */
export interface BlockDocument {
  schemaVersion: number;
  id: string;
  title: string;
  blocks: Block[];
}

/** 読み込んだ文書が現行スキーマかどうか。false なら移行処理が必要。 */
export function isCurrentSchema(doc: BlockDocument): boolean {
  return doc.schemaVersion === SCHEMA_VERSION;
}
