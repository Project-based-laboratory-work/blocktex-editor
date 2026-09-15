import { describe, expect, it } from 'vitest';
import { SCHEMA_VERSION, generateId, type BlockDocument } from './index';

describe('model', () => {
  it('generateId returns unique string ids', () => {
    const a = generateId();
    const b = generateId();
    expect(a).not.toBe(b);
    expect(typeof a).toBe('string');
  });

  it('builds a BlockDocument carrying the current schema version', () => {
    const doc: BlockDocument = {
      schemaVersion: SCHEMA_VERSION,
      id: generateId(),
      title: 'サンプル',
      blocks: [{ id: generateId(), type: 'heading', level: 1, text: '見出し' }],
    };

    expect(doc.schemaVersion).toBe(1);
    expect(doc.blocks[0].type).toBe('heading');
  });
});
