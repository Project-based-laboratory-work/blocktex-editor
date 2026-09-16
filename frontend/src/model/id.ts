/**
 * ブロック・文書のID生成。
 *
 * crypto.randomUUID() は secure context（https または localhost）でしか定義されない。
 * `npm run dev -- --host` で立てた開発サーバに http://192.168.x.x:5173 のような
 * LAN内アドレスでアクセスすると undefined になり、ブロック追加時に落ちる。
 * チームでの動作確認や発表デモで踏みうるため、フォールバックを用意する。
 */
export function generateId(): string {
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
