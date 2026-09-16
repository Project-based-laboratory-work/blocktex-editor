import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

// vite.config.ts で globals:false にしているため afterEach がグローバルに存在せず、
// Testing Library の自動クリーンアップが働かない（render したDOMが次のテストに残る）。
// 明示的に登録しておく。
afterEach(() => {
  cleanup();
});
