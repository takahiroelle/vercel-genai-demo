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
