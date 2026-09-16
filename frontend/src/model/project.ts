import type { Block } from './block';
import type { SCHEMA_VERSION } from './document';

// Firestore の users/{uid}/projects/{projectId} 1件に対応する。
// Firestore の Timestamp ↔ Date の変換はPhase3でSDKとの境界で行う。
export interface CloudProject {
  id: string;
  name: string;
  schemaVersion: typeof SCHEMA_VERSION;
  blocks: Block[];
  createdAt: Date;
  updatedAt: Date;
}
