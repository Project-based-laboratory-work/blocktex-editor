import type { BlockDocument } from '../model';
import sampleDocumentJson from './sample-document.json';

/**
 * サンプル文書を BlockDocument 型で読むための入口。
 *
 * JSONモジュールを import するとTypeScriptは文字列リテラルを string に広げてしまい
 * （`type: 'heading'` が `type: string` になる）、`.json` を直接 BlockDocument には代入できない。
 * fixtureは中身をこちらで管理しているデータなので、ここで一度だけ型を確定させ、
 * 利用側（Phase1のスナップショットテスト等）ではキャストを書かなくて済むようにする。
 */
export const sampleDocument = sampleDocumentJson as BlockDocument;
