import type { Block } from './block';

/**
 * Firestore の users/{uid}/projects/{projectId} 1件に対応する（Phase3）。
 *
 * BlockDocument との対応:
 *   CloudProject.id   ← FirestoreのドキュメントID。ドキュメント本体のフィールドには持たせない
 *   CloudProject.name ← BlockDocument.title と同じもの。Firestore側は進捗シート3-3-3の
 *                       「名前変更」に合わせて name と呼ぶ
 *   blocks            ← BlockDocument.blocks をそのまま（ローカル保存と同じ形式）
 *
 * Firestore の Timestamp ↔ Date の変換はPhase3でSDKとの境界で行う。
 */
export interface CloudProject {
  id: string;
  name: string;
  schemaVersion: number;
  blocks: Block[];
  createdAt: Date;
  updatedAt: Date;
}
