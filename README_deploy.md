# Streamlit NN 可視化アプリ Vercel デプロイ手順（Docker）

含まれるファイル:

- app_streamlit_nodes_labeled_parallel_full.py
- common_font_wsl.py
- requirements.txt
- Dockerfile
- vercel.json

## ローカル確認

pip install -r requirements.txt
streamlit run app_streamlit_nodes_labeled_parallel_full.py

## デプロイ

vercel
vercel --prod

## AWS EC2 (CDK + ECR コンテナ) デプロイ

WebSocket が必須のため、AWS Copilot/ECS/Fargate の代わりに EC2 へ直接デプロイする CDK スタックを追加しています。すでに ECR に `streamlit-viz` イメージ（タグ `latest` など）が push 済みである前提です。

1. 依存ライブラリをインストール
   ```bash
   cd cdk
   npm install
   ```
2. CDK をブートストラップ（アカウント/リージョンごとに 1 回）
   ```bash
   npx cdk bootstrap aws://<ACCOUNT_ID>/<REGION>
   ```
3. デプロイ（context で ECR やインスタンスタイプ、HTTPS 設定を指定可能）
   ```bash
   npm run deploy -- \\
     -c repoName=streamlit-viz \\
     -c imageTag=latest \\
     -c containerPort=8000 \\
     -c instanceType=t3.small \\
     -c hostedZoneDomain=mananda.org \\
     -c subdomain=genai
   ```
   - `repoName`: 既存の ECR リポジトリ名
   - `imageTag`: 使用するタグ（省略時 `latest`）
   - `containerPort`: `Dockerfile` の `ENV PORT` に合わせる（デフォルト 8000）
   - `instanceType`: `-c instanceType=t3.micro` のように指定可能（デフォルト `t3.small`）
   - `hostedZoneDomain` / `subdomain`: Route 53 にホストゾーンが存在する場合、CDK が DNS 検証付き ACM 証明書と A レコード (`genai.mananda.org`) を自動作成し、HTTPS(443) を終端します。
   - `certificateArn`: 既存証明書を手動指定したい場合はこちらを使用（`hostedZoneDomain` を省略すると `certificateArn` を読む挙動です）。
4. デプロイ完了後、スタック出力の `CustomDomainUrl` もしくは `LoadBalancerDnsName` にアクセスすると HTTPS/HTTP で利用できます。トラブルシュートや内部確認用に、`PublicDns`/`PublicIp` の `:8000` へ直接アクセスすることも可能です。CloudFormation スタックを削除すればリソース一式がクリーンアップされます。
   - EC2 コンテナの標準出力は CloudWatch Logs (`/streamlit/StreamlitEc2Stack`) に送信されます。`aws logs tail /streamlit/StreamlitEc2Stack --follow` で `[font] ...` ログなどを確認できます。
