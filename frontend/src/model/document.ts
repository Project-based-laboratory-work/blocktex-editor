import type { Block } from './block';

export const SCHEMA_VERSION = 1 as const;

export interface BlockDocument {
  schemaVersion: typeof SCHEMA_VERSION;
  id: string;
  title: string;
  blocks: Block[];
}
