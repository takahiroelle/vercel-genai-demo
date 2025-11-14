FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV LOCAL_JP_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc

CMD streamlit run app_streamlit_nodes_labeled_parallel_full.py \
    --server.port ${PORT} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
