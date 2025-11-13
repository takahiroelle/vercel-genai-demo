# app_streamlit_nodes_labeled_parallel_full.py
# pip install streamlit numpy matplotlib pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import cm
from common_font_wsl import set_font_auto

st.set_page_config(page_title="Node Viz (Full, Labeled, Compare + Diff)", layout="wide")
st.title("ノード可視化：未学習→学習→学習後／ヒートマップ比較（差分）／並列比較（ラベル付き）")

manual = st.sidebar.text_input("フォントファイルパス（任意）", value="")
chosen = set_font_auto(manual_path=manual, bundled_dir="fonts")
st.sidebar.write("使用フォント:", chosen or "未設定（英数字のみ）")

vocab = ["猫","犬","学校","先生","AI","学習","記憶","重み","データ","モデル","推論","最適化","誤差","勾配","ニューラル","層"]
V = len(vocab)

with st.sidebar:
    H = st.slider("隠れ層サイズ H", 8, 128, 24, 1)
    lr = st.slider("学習率", 0.01, 1.0, 0.8, 0.01)
    epochs = st.slider("エポック", 50, 3000, 800, 50)
    seed = st.number_input("乱数シード", min_value=0, value=42, step=1)
    noise = st.slider("ノイズ σ（学習後）", 0.0, 0.5, 0.10, 0.01)
    topk = st.slider("隠れ層 Top-k ハイライト", 0, 32, 5, 1)
    prob_edges = st.checkbox("出力確率で Hidden→Output のエッジ色を変える", value=True)
    show_labels = st.checkbox("入出力ノードにラベルを表示（日本語）", value=True)
    separate_norm = st.checkbox("ヒートマップの Before/After カラースケールを分離", value=False)
    diff_cmap = st.selectbox("差分ヒートマップの色", ["bwr","seismic","coolwarm","RdBu","PiYG"], index=0)

rng = np.random.default_rng(seed)
X = np.eye(V); T = np.eye(V)
W1 = rng.normal(0, 0.2, size=(V, H)); b1 = np.zeros(H)
W2 = rng.normal(0, 0.2, size=(H, V)); b2 = np.zeros(V)

def forward(x, W1, b1, W2, b2):
    h = np.tanh(x @ W1 + b1)
    z = h @ W2 + b2
    z -= z.max()
    y = np.exp(z)/np.exp(z).sum()
    return h, y, z

def train(W1, b1, W2, b2, lr, epochs):
    losses = []
    for _ in range(epochs):
        dW1 = np.zeros_like(W1); db1 = np.zeros_like(b1)
        dW2 = np.zeros_like(W2); db2 = np.zeros_like(b2)
        L = 0.0
        for i in range(V):
            x, t = X[i], T[i]
            h, y, _ = forward(x, W1, b1, W2, b2)
            L += -np.log(y[i] + 1e-12)
            dz = y - t
            dW2 += np.outer(h, dz); db2 += dz
            dh = dz @ W2.T
            dh_raw = (1 - h**2) * dh
            dW1 += np.outer(x, dh_raw); db1 += dh_raw
        L/=V; dW1/=V; db1/=V; dW2/=V; db2/=V
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
        losses.append(L)
    return W1, b1, W2, b2, np.array(losses)

def draw_network(W1, W2, vocab, active_input_idx=None, hidden_act=None, y=None, title="", topk=0, prob_edges=True, show_labels=True):
    V = len(vocab); H = W1.shape[1]
    x_in, x_h, x_out = 0.0, 0.5, 1.0
    y_in = np.linspace(0,1,V); y_h = np.linspace(0,1,H); y_out = np.linspace(0,1,V)
    wmax = max(np.abs(W1).max(), np.abs(W2).max())
    fig = plt.figure(figsize=(9.8,5.6))
    plt.title(title)

    in_colors = ["red" if (active_input_idx is not None and i == active_input_idx) else "lightgray" for i in range(V)]
    plt.scatter([x_in]*V, y_in, s=90, c=in_colors, edgecolors="black", zorder=3)

    if hidden_act is not None:
        norm = (hidden_act - hidden_act.min()) / (hidden_act.max() - hidden_act.min() + 1e-12)
        hidden_rgba = cm.YlOrRd(norm)
        top_idx = np.argsort(-np.abs(hidden_act))[:max(0, min(topk, len(hidden_act)))]
        plt.scatter([x_h]*H, y_h, s=80, c=hidden_rgba, edgecolors="black", zorder=3)
        if topk > 0 and len(top_idx) > 0:
            plt.scatter([x_h]*len(top_idx), y_h[top_idx], s=160, facecolors="none", edgecolors="orange", linewidths=2.0, zorder=4)
    else:
        plt.scatter([x_h]*H, y_h, s=80, c="lightgray", edgecolors="black", zorder=3)

    if y is not None:
        out_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
        out_rgba = cm.Blues(out_norm)
        argmax_idx = int(np.argmax(y))
    else:
        out_rgba = ["lightgray"]*V
        argmax_idx = None
    plt.scatter([x_out]*V, y_out, s=90, c=out_rgba, edgecolors="black", zorder=3)
    if argmax_idx is not None:
        plt.scatter([x_out], [y_out[argmax_idx]], s=160, facecolors="none", edgecolors="blue", linewidths=2.0, zorder=4)

    for i in range(V):
        for j in range(H):
            lw = 0.5 + 4.0*(abs(W1[i,j])/(wmax+1e-12))
            color = "red" if (active_input_idx is not None and i == active_input_idx) else "gray"
            plt.plot([x_in, x_h], [y_in[i], y_h[j]], linewidth=lw, alpha=0.5, color=color)
    for j in range(H):
        for k in range(V):
            lw = 0.5 + 4.0*(abs(W2[j,k])/(wmax+1e-12))
            if prob_edges and y is not None:
                c = cm.Blues((y[k] - y.min()) / (y.max() - y.min() + 1e-12))
                plt.plot([x_h, x_out], [y_h[j], y_out[k]], linewidth=lw, alpha=0.35, color=c)
            else:
                plt.plot([x_h, x_out], [y_h[j], y_out[k]], linewidth=lw, alpha=0.35, color="gray")

    if show_labels:
        for i in range(V):
            plt.text(x_in - 0.05, y_in[i], vocab[i], ha="right", va="center", fontsize=9)
        for k in range(V):
            plt.text(x_out + 0.05, y_out[k], vocab[k], ha="left", va="center", fontsize=9)

    plt.xticks([x_in,x_h,x_out], ["Input","Hidden","Output"])
    plt.yticks([]); plt.xlim(-0.18,1.18); plt.ylim(-0.05,1.05)
    return fig

tabs = st.tabs(["未学習の可視化", "学習", "学習後の可視化", "重みヒートマップ比較（差分付き）", "ネットワーク図（並列比較）"])
tab_before, tab_train, tab_after, tab_wcmp, tab_graph = tabs

with tab_before:
    st.subheader("未学習の入出力とノード（ラベル付き）")
    word = st.selectbox("単語を選択（未学習）", vocab, key="before_word")
    idx = vocab.index(word)
    x = X[idx]
    h, y, z = forward(x, W1, b1, W2, b2)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**入力ベクトル（one-hot）**")
        st.code(np.array2string(x, precision=0, separator=", "), language="text")
        st.markdown("**未学習の出力確率**")
        df_y = pd.DataFrame({"word": vocab, "p": y})
        st.dataframe(df_y.style.format({"p": "{:.4f}"}), use_container_width=True)
        st.markdown(f"**未学習の予測:** {vocab[int(np.argmax(y))]}  （確信度 {y.max():.3f}）")
        fig_out = plt.figure()
        plt.bar(np.arange(V), y); plt.xticks(np.arange(V), vocab, rotation=45, ha="right"); plt.ylim(0,1)
        st.pyplot(fig_out, clear_figure=True)
    with c2:
        st.markdown("**隠れ層ノードの活性（tanh）**")
        fig_h = plt.figure()
        plt.bar(np.arange(len(h)), h); plt.xlabel("Hidden index"); plt.ylabel("Activation")
        st.pyplot(fig_h, clear_figure=True)

    st.markdown("**ネットワーク図（未学習）**")
    st.pyplot(draw_network(W1, W2, vocab, active_input_idx=idx, hidden_act=h, y=y,
                           title=f"未学習：{word}", topk=topk, prob_edges=prob_edges, show_labels=show_labels), clear_figure=True)

with tab_train:
    st.subheader("学習（LossとBeforeスナップショット保存）")
    if st.button("学習スタート"):
        st.session_state["W1_before"] = W1.copy()
        st.session_state["W2_before"] = W2.copy()
        W1, b1, W2, b2, losses = train(W1, b1, W2, b2, lr, epochs)
        st.session_state["trained_params"] = (W1, b1, W2, b2)
        st.line_chart(losses, height=220)
        st.success(f"最終損失: {losses[-1]:.6f}")
    else:
        st.info("サイドバーを調整して『学習スタート』を押してください。")

with tab_after:
    st.subheader("学習後の入出力とノード（ラベル付き）")
    if "trained_params" not in st.session_state:
        st.warning("『学習』タブで学習を実行してください。")
    else:
        W1_t, b1_t, W2_t, b2_t = st.session_state["trained_params"]
        word2 = st.selectbox("単語を選択（学習後）", vocab, key="after_word")
        idx2 = vocab.index(word2)
        x2 = X[idx2] + rng.normal(0, noise, size=X[idx2].shape)
        h2, y2, z2 = forward(x2, W1_t, b1_t, W2_t, b2_t)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**入力ベクトル（ノイズ付）**")
            st.code(np.array2string(x2, precision=2, separator=", "), language="text")
            st.markdown("**学習後の出力確率**")
            df_y2 = pd.DataFrame({"word": vocab, "p": y2})
            st.dataframe(df_y2.style.format({"p": "{:.4f}"}), use_container_width=True)
            st.markdown(f"**学習後の予測:** {vocab[int(np.argmax(y2))]}  （確信度 {y2.max():.3f}）")
            fig_out2 = plt.figure()
            plt.bar(np.arange(V), y2); plt.xticks(np.arange(V), vocab, rotation=45, ha="right"); plt.ylim(0,1)
            st.pyplot(fig_out2, clear_figure=True)
        with c2:
            st.markdown("**隠れ層ノードの活性（学習後）**")
            fig_h2 = plt.figure()
            plt.bar(np.arange(len(h2)), h2); plt.xlabel("Hidden index"); plt.ylabel("Activation")
            st.pyplot(fig_h2, clear_figure=True)

        st.markdown("**ネットワーク図（学習後）**")
        st.pyplot(draw_network(W1_t, W2_t, vocab, active_input_idx=idx2, hidden_act=h2, y=y2,
                               title=f"学習後：{word2}", topk=topk, prob_edges=prob_edges, show_labels=show_labels), clear_figure=True)

with tab_wcmp:
    st.subheader("重みヒートマップ Before/After 比較（差分付き）")
    if "trained_params" not in st.session_state or "W1_before" not in st.session_state:
        st.warning("『学習』タブで学習を実行してください（Beforeスナップショット保存）。")
    else:
        W1_bef = st.session_state["W1_before"]
        W2_bef = st.session_state["W2_before"]
        W1_aft, _, W2_aft, _ = st.session_state["trained_params"]

        if separate_norm:
            vmin1_b, vmax1_b = W1_bef.min(), W1_bef.max()
            vmin2_b, vmax2_b = W2_bef.min(), W2_bef.max()
            vmin1_a, vmax1_a = W1_aft.min(), W1_aft.max()
            vmin2_a, vmax2_a = W2_aft.min(), W2_aft.max()
        else:
            vmin1 = min(W1_bef.min(), W1_aft.min()); vmax1 = max(W1_bef.max(), W1_aft.max())
            vmin2 = min(W2_bef.min(), W2_aft.min()); vmax2 = max(W2_bef.max(), W2_aft.max())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Before**")
            fig1 = plt.figure(); plt.imshow(W1_bef, aspect="auto", vmin=(vmin1_b if separate_norm else vmin1), vmax=(vmax1_b if separate_norm else vmax1)); plt.colorbar(); plt.title("W1 (Before)"); plt.xlabel("Hidden"); plt.ylabel("入力（単語）"); plt.yticks(np.arange(V), vocab); st.pyplot(fig1, clear_figure=True)
            fig2 = plt.figure(); plt.imshow(W2_bef, aspect="auto", vmin=(vmin2_b if separate_norm else vmin2), vmax=(vmax2_b if separate_norm else vmax2)); plt.colorbar(); plt.title("W2 (Before)"); plt.xlabel("出力（単語）"); plt.ylabel("Hidden"); st.pyplot(fig2, clear_figure=True)
        with c2:
            st.markdown("**After**")
            fig3 = plt.figure(); plt.imshow(W1_aft, aspect="auto", vmin=(vmin1_a if separate_norm else vmin1), vmax=(vmax1_a if separate_norm else vmax1)); plt.colorbar(); plt.title("W1 (After)"); plt.xlabel("Hidden"); plt.ylabel("入力（単語）"); plt.yticks(np.arange(V), vocab); st.pyplot(fig3, clear_figure=True)
            fig4 = plt.figure(); plt.imshow(W2_aft, aspect="auto", vmin=(vmin2_a if separate_norm else vmin2), vmax=(vmax2_a if separate_norm else vmax2)); plt.colorbar(); plt.title("W2 (After)"); plt.xlabel("出力（単語）"); plt.ylabel("Hidden"); st.pyplot(fig4, clear_figure=True)
        with c3:
            st.markdown("**Diff (After − Before)**")
            diff1 = W1_aft - W1_bef; diff2 = W2_aft - W2_bef
            d1max = np.max(np.abs(diff1)); d2max = np.max(np.abs(diff2))
            cmap = diff_cmap
            fig5 = plt.figure(); plt.imshow(diff1, aspect="auto", vmin=-d1max, vmax=d1max, cmap=cmap); plt.colorbar(); plt.title("ΔW1"); plt.xlabel("Hidden"); plt.ylabel("入力（単語）"); plt.yticks(np.arange(V), vocab); st.pyplot(fig5, clear_figure=True)
            fig6 = plt.figure(); plt.imshow(diff2, aspect="auto", vmin=-d2max, vmax=d2max, cmap=cmap); plt.colorbar(); plt.title("ΔW2"); plt.xlabel("出力（単語）"); plt.ylabel("Hidden"); st.pyplot(fig6, clear_figure=True)

with tab_graph:
    st.subheader("ネットワーク図（並列比較・ラベル付き）")
    if "trained_params" not in st.session_state:
        st.warning("『学習』タブで学習を実行してください。")
    else:
        W1_b, W2_b = st.session_state["W1_before"], st.session_state["W2_before"]
        W1_a, b1_a, W2_a, b2_a = st.session_state["trained_params"]
        word_cmp = st.selectbox("比較する単語", vocab, index=0, key="cmp_word")
        idx_cmp = vocab.index(word_cmp)

        # Before 用の隠れ層次元は W1_b から計算する
        H_b = W1_b.shape[1]

        x_b = X[idx_cmp]
        h_b, y_b, _ = forward(x_b, W1_b, np.zeros(H_b), W2_b, np.zeros(V))

        x_a = X[idx_cmp] + rng.normal(0, noise, size=X[idx_cmp].shape)
        h_a, y_a, _ = forward(x_a, W1_a, b1_a, W2_a, b2_a)

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(draw_network(W1_b, W2_b, vocab, active_input_idx=idx_cmp, hidden_act=h_b, y=y_b,
                                   title=f"Before（{word_cmp}）", topk=topk, prob_edges=prob_edges, show_labels=show_labels), clear_figure=True)
        with c2:
            st.pyplot(draw_network(W1_a, W2_a, vocab, active_input_idx=idx_cmp, hidden_act=h_a, y=y_a,
                                   title=f"After（{word_cmp}）", topk=topk, prob_edges=prob_edges, show_labels=show_labels), clear_figure=True)
