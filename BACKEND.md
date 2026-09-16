# バックエンド設計（Phase 0 / 0-5）

Phase 3「アカウント・クラウド保存機能」に向けた設計。実装はPhase 3で行い、Phase 0では方針の確定までを行う。

## 0-5-1 Firebase / Supabase の比較検討・選定

| 観点 | Firebase (Firestore) | Supabase (Postgres) |
|---|---|---|
| データモデル | ドキュメント型。ブロック配列のJSONをそのまま1ドキュメントに保存できる | リレーショナル。JSONはjsonb列に入れる形になる |
| 認証 | Firebase Authでメール/Googleログインが数行で導入できる | Supabase Authで同等のことができる |
| アクセス制御 | セキュリティルール（独自DSL） | RLSポリシー（SQL） |
| 無料枠の運用 | 放置してもプロジェクトが停止しない | 無料プランは一定期間未使用でプロジェクトが一時停止し、復帰操作が必要 |
| ホスティング | Firebase Hostingが同一コンソール・同一CLIに統合 | 別途Vercel等を用意する |
| 日本語の情報量 | 多い | Firebaseよりは少ない |

**選定結果: Firebase を採用する。**

理由:

1. 保存したいデータがブロック配列のJSONそのもので、ドキュメント型のFirestoreと形が一致する。リレーショナルな検索要件が無いため、SQLの利点を活かす場面が無い。
2. 無料プランでもプロジェクトが自動停止しない。本プロジェクトは発表（10/7・11/4・12/16）の間隔が空くため、久しぶりに触ったら止まっていた、という事故を避けたい。
3. 認証・DB・ホスティングが1つのコンソールとCLIにまとまり、5人のチームでセットアップ手順を共有する負担が最小になる。

## 0-5-2 認証方式の設計

**Firebase Auth を使い、Googleログインを主、メール/パスワードを従とする。**

- **Googleログイン（主）**: ワンクリックでログインでき、こちらでパスワードを保持・管理しなくて済む。大学のGoogleアカウントがそのまま使える。
- **メール/パスワード（従）**: Googleアカウントを使いたくない場合の代替。実装コストが低いので併設する。

認証状態の扱い:

- ログイン中ユーザーは `onAuthStateChanged` で監視し、アプリ全体で保持する（Phase 3 / 3-2-2）。
- 以降のFirestoreアクセスはすべてログインユーザーの `uid` をパスに含める形で行い、他人のデータには構造的に触れられないようにする。
- 未ログイン状態でもブロックエディタ自体は使える（ローカル保存はPhase 2で実装済みの想定）。クラウド保存だけをログイン必須にする。

## 0-5-3 保存データのスキーマ設計

Firestoreのコレクション構成:

```
users/{uid}/projects/{projectId}
  name          : string     プロジェクト名（ユーザーが変更可能）
  schemaVersion : number     ブロックデータのスキーマ版数（frontend/src/model の SCHEMA_VERSION）
  blocks        : Block[]    ブロック配列（ローカル保存と同じ形式）
  createdAt     : Timestamp
  updatedAt     : Timestamp
```

設計の意図:

- **`users/{uid}` 配下のサブコレクションにする**: セキュリティルールが「パスのuidとログインユーザーのuidが一致するか」の1条件で済み、他人のデータが見えない保証が構造で担保される。一覧取得もこのコレクションを読むだけで済む（Phase 3 / 3-3-2）。
- **`blocks` はローカル保存と同じ形式にする**: ローカル保存（Phase 2）とクラウド保存（Phase 3）で変換処理を二重に書かずに済む。`schemaVersion` を一緒に保存しておくことで、後からスキーマを変えたときに移行処理を書ける。
- **プロジェクト名と本文を同じドキュメントに入れる**: 一覧画面で名前だけ欲しい場合に本文も読み込むことになるが、この規模（1人あたり数件〜数十件）ではドキュメント数を増やす方が読み取り回数の面で不利になる。

想定するセキュリティルール（実装はPhase 3 / 3-1-3）:

```
match /users/{uid}/{document=**} {
  allow read, write: if request.auth != null && request.auth.uid == uid;
}
```

**注意すべき制約**: Firestoreの1ドキュメントは最大1MiB。画像をBase64で埋め込むとすぐに超えるため、画像はFirebase Storageに置いてURLだけをブロックに持たせる方式を推奨する。Phase 2の「画像データの扱い方針決定」（2-2-1）は、この制約を踏まえて決めること。

## 0-5-4 初期セットアップ手順

Firebaseプロジェクトの作成は各自のGoogleアカウントとブラウザ操作が必要なため、**チームの誰か1人が以下を実施する**（未実施）。

1. [Firebase コンソール](https://console.firebase.google.com/) でプロジェクトを作成する（Google Analyticsは不要）。
2. 「ウェブアプリを追加」して、表示される設定値（apiKey, authDomain, projectId など）を控える。
3. Authentication を有効化し、ログイン方法で「Google」と「メール/パスワード」を有効にする。
4. Firestore Database を作成する（本番モードで作成し、ルールはPhase 3で設定する）。
5. 控えた設定値を `frontend/.env`（`frontend/.env.example` をコピーして作成）に記入する。`.env` はgitignore済みなのでコミットされない。
6. 無料枠（Spark プラン）の内容を[料金ページ](https://firebase.google.com/pricing)で確認する。読み取り・書き込み回数やストレージに1日あたりの上限があるが、本プロジェクトの規模（5人での開発とデモ）であれば通常は上限に達しない。**上限値は変更されることがあるため、必ず公式ページで最新の値を確認すること。**

## 0-5-5 フロントエンドのホスティング先

**Firebase Hosting を採用する。** バックエンドと同じコンソール・同じCLI（`firebase`）で完結し、アカウントを増やさずに済むため。

デプロイ手順（Phase 3以降）:

```bash
cd frontend
npm run build              # dist/ を生成
firebase deploy --only hosting
```

`firebase init hosting` 実行時の設定:

- public directory: `frontend/dist`
- single-page app: Yes（Reactのルーティングのため）
- GitHub連携の自動デプロイ: 任意
