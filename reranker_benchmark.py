# import marimo

# app = marimo.App(width="full")

# @app.cell
# def __():
#     import math
#     import os
#     import time
#     import altair as alt
#     from datasets import load_dataset
#     import marimo as mo
#     import pandas as pd
#     import numpy as np
#     from sentence_transformers import CrossEncoder

#     try:
#         from jev_reranker import JevReranker
#     except ImportError:
#         JevReranker = None

#     return (
#         CrossEncoder,
#         JevReranker,
#         alt,
#         load_dataset,
#         math,
#         mo,
#         np,
#         os,
#         pd,
#         time,
#     )

# @app.cell
# def __(mo):
#     header_view = mo.md(
#         """
#         # Reranker Benchmark: Qwen3-Reranker vs Jev (for now)

#         This reactive benchmark evaluates **semantic ranking quality** and **inference latency**
#         """
#     )
#     return header_view

# @app.cell
# def __(header_view):
#     header_view
#     return

# @app.cell
# def __(mo):
#     jev_key_input = mo.ui.text(
#         label = "Jev API Key",
#         kind = "password",
#         placeholder = "Enter your Jev API key here",
#     )

#     model_choice = mo.ui.dropdown(
#         options=[
#             "Qwen/Qwen3-Reranker-0.6b"
#         ],
#         value = "Qwen/Qwen3-Reranker-0.6b",
#         label = "Qwen3 Model variant",
#     )

#     query_count_slider = mo.ui.slider(
#         start = 5,
#         stop = 50,
#         step = 5,
#         value = 10,
#         label = "Queries to Evaluate",
#     )

#     candidates_slider = mo.ui.slider(
#              start=10,
#         stop=60,
#         step=10,
#         value=30,
#         label="Candidates per Query",
#     )

#     threshold_slider = mo.ui.slider(
#         start=0.0,
#         stop=0.5,
#         step=0.05,
#         value=0.0,
#         label="Jev Cutoff Threshold",
#     )

#     run_button = mo.ui.run_button(label = "Execute Bitch")

#     settings_panel = mo.vstack([
#         mo.md("### Benchmark Config"),
#         mo.hstack([jev_key_input, model_choice]),
#         mo.hstack([query_count_slider, candidates_slider, threshold_slider]),
#         run_button,
#     ])

#     return (
#         candidates_slider,
#         jev_key_input,
#         model_choice,
#         query_count_slider,
#         run_button,
#         settings_panel,
#         threshold_slider,
#     )

# @app.cell
# def __(settings_panel):
#     settings_panel
#     return

# @app.cell
# def __(candidates_slider, load_dataset, mo, query_count_slider):
#     mo.md("Loading 'sentence-transformers/NanoHotpotQA-bm25"),

import marimo

app = marimo.App(width="full")


@app.cell
def __():
    import math
    import os
    import time
    import altair as alt
    from datasets import load_dataset
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sentence_transformers import CrossEncoder

    # Graceful import check for Jev client library
    try:
        from jev_reranker import JevReranker
    except ImportError:
        JevReranker = None

    return (
        CrossEncoder,
        JevReranker,
        alt,
        load_dataset,
        math,
        mo,
        np,
        os,
        pd,
        time,
    )


@app.cell
def __(mo):
    header_view = mo.md(
        """
        # 🥊 Reranker Evaluation: Qwen3-Reranker vs. Jev (TypeSafe)

        This reactive benchmark evaluates **semantic ranking quality** and **inference latency**
        on the **NanoHotpotQA** benchmark dataset.

        - **Qwen3-Reranker**: Causal pointwise cross-encoder model running locally.
        - **Jev Reranker**: Remote listwise steerable ranking API with semantic filtering.
        """
    )
    return header_view,


@app.cell
def __(header_view):
    header_view
    return


@app.cell
def __(mo):
    jev_key_input = mo.ui.text(
        label="Jev API Key",
        kind="password",
        placeholder="Enter your TypeSafe API key (or set JEV_API_KEY env var)",
    )

    model_choice = mo.ui.dropdown(
        options=[
            "Qwen/Qwen3-Reranker-0.6B",
            "Qwen/Qwen3-Reranker-4B",
        ],
        value="Qwen/Qwen3-Reranker-0.6B",
        label="Qwen3 Model Variant",
    )

    query_count_slider = mo.ui.slider(
        start=5,
        stop=50,
        step=5,
        value=15,
        label="Queries to Evaluate",
    )

    candidates_slider = mo.ui.slider(
        start=10,
        stop=60,
        step=10,
        value=30,
        label="Candidates per Query",
    )

    threshold_slider = mo.ui.slider(
        start=0.0,
        stop=0.5,
        step=0.05,
        value=0.0,
        label="Jev Cutoff Threshold",
    )

    run_btn = mo.ui.run_button(label="🚀 Execute Benchmark")

    # Render settings card
    settings_panel = mo.vstack([
        mo.md("### ⚙️ Benchmark Configuration"),
        mo.hstack([jev_key_input, model_choice]),
        mo.hstack([query_count_slider, candidates_slider, threshold_slider]),
        run_btn,
    ])

    return (
        candidates_slider,
        jev_key_input,
        model_choice,
        query_count_slider,
        run_btn,
        settings_panel,
        threshold_slider,
    )


@app.cell
def __(settings_panel):
    settings_panel
    return


@app.cell
def __(candidates_slider, load_dataset, mo, query_count_slider):
    mo.md("⏳ Loading `sentence-transformers/NanoHotpotQA-bm25`...")

    ds_corpus = load_dataset(
        "sentence-transformers/NanoHotpotQA-bm25", split="corpus"
    )
    ds_queries = load_dataset(
        "sentence-transformers/NanoHotpotQA-bm25", split="queries"
    )
    ds_qrels = load_dataset(
        "sentence-transformers/NanoHotpotQA-bm25", split="qrels"
    )

    # Build corpus mapping: doc_id -> text
    corpus_lookup = {row["_id"]: row["text"] for row in ds_corpus}

    # Build ground truth dictionary: query_id -> {doc_id: relevance_score}
    ground_truth = {}
    for row in ds_qrels:
        qid = row["query-id"]
        cid = row["corpus-id"]
        score = row.get("score", 1)
        ground_truth.setdefault(qid, {})[cid] = score

    # Assemble query pool with relevant candidate docs
    eval_dataset = []
    num_queries = query_count_slider.value
    max_candidates = candidates_slider.value

    for query_item in ds_queries.select(range(min(num_queries, len(ds_queries)))):
        qid = query_item["_id"]
        qtext = query_item["text"]
        relevant_docs = ground_truth.get(qid, {})

        # Extract top BM25 candidates
        candidate_ids = query_item.get("bm25-ranked-ids", [])[:max_candidates]

        # Ensure known ground-truth documents are included in the pool
        for gt_id in relevant_docs:
            if gt_id not in candidate_ids and gt_id in corpus_lookup:
                candidate_ids.append(gt_id)

        doc_pairs = [
            (cid, corpus_lookup[cid])
            for cid in candidate_ids
            if cid in corpus_lookup
        ]

        if doc_pairs:
            eval_dataset.append({
                "qid": qid,
                "query": qtext,
                "docs": doc_pairs,
                "qrels": relevant_docs,
            })

    status_message = mo.md(
        f"✅ Loaded **{len(eval_dataset)}** queries with up to **{max_candidates}** candidate passages each."
    )
    return (
        candidate_ids,
        corpus_lookup,
        doc_pairs,
        ds_corpus,
        ds_qrels,
        ds_queries,
        eval_dataset,
        ground_truth,
        max_candidates,
        num_queries,
        qid,
        qtext,
        query_item,
        relevant_docs,
        status_message,
    )


@app.cell
def __(status_message):
    status_message
    return


@app.cell
def __(math):
    def evaluate_ranking(ranked_ids, qrels, k=10):
        """Calculates nDCG@k and MRR@k against binary/graded ground truth."""
        # 1. Discounted Cumulative Gain (DCG@k)
        dcg = 0.0
        for rank, doc_id in enumerate(ranked_ids[:k], start=1):
            rel = qrels.get(doc_id, 0)
            if rel > 0:
                dcg += rel / math.log2(rank + 1)

        # 2. Ideal Discounted Cumulative Gain (IDCG@k)
        sorted_rels = sorted(qrels.values(), reverse=True)[:k]
        idcg = sum(
            rel / math.log2(rank + 1)
            for rank, rel in enumerate(sorted_rels, start=1)
        )
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        # 3. Mean Reciprocal Rank (MRR@k)
        mrr = 0.0
        for rank, doc_id in enumerate(ranked_ids[:k], start=1):
            if qrels.get(doc_id, 0) > 0:
                mrr = 1.0 / rank
                break

        return {"ndcg@10": ndcg, "mrr@10": mrr}

    return evaluate_ranking,


@app.cell
def __(
    CrossEncoder,
    JevReranker,
    eval_dataset,
    evaluate_ranking,
    jev_key_input,
    model_choice,
    np,
    os,
    run_btn,
    threshold_slider,
    time,
):
    benchmark_records = []
    qwen_metrics_list = []
    jev_metrics_list = []
    has_run = run_btn.value

    if has_run:
        # Resolve Jev API key from input box or environment
        effective_key = jev_key_input.value or os.environ.get("JEV_API_KEY", "")

        # Initialize local Qwen Cross-Encoder
        qwen_model = CrossEncoder(model_choice.value, trust_remote_code=True)

        # Initialize Jev client if available and configured
        jev_client = None
        if effective_key and JevReranker:
            try:
                jev_client = JevReranker(api_key=effective_key)
            except Exception:
                jev_client = None

        cutoff_threshold = threshold_slider.value

        for sample in eval_dataset:
            query = sample["query"]
            docs = sample["docs"]
            qrels = sample["qrels"]

            doc_ids = [d[0] for d in docs]
            doc_texts = [d[1] for d in docs]

            # --- Evaluate Qwen3-Reranker ---
            qwen_t0 = time.perf_counter()
            query_doc_pairs = [(query, text) for text in doc_texts]
            qwen_scores = qwen_model.predict(query_doc_pairs, batch_size=32)
            qwen_latency_ms = (time.perf_counter() - qwen_t0) * 1000

            # Rank by descending cross-encoder score
            qwen_order = np.argsort(qwen_scores)[::-1]
            qwen_ranked_ids = [doc_ids[idx] for idx in qwen_order]
            qwen_eval = evaluate_ranking(qwen_ranked_ids, qrels, k=10)

            # --- Evaluate Jev Reranker ---
            jev_latency_ms = 0.0
            jev_eval = {"ndcg@10": None, "mrr@10": None}
            jev_retained_count = len(doc_texts)

            if jev_client:
                jev_t0 = time.perf_counter()
                try:
                    response = jev_client.relevance_rerank(
                        query=query,
                        documents=doc_texts,
                        threshold=cutoff_threshold,
                    )
                    jev_latency_ms = (time.perf_counter() - jev_t0) * 1000

                    # Map returned passages back to IDs
                    text_to_id = {text: did for did, text in docs}
                    jev_ranked_ids = [
                        text_to_id.get(item["text"])
                        for item in response.get("results", [])
                        if item.get("text") in text_to_id
                    ]
                    jev_retained_count = len(jev_ranked_ids)
                    jev_eval = evaluate_ranking(jev_ranked_ids, qrels, k=10)
                except Exception:
                    jev_latency_ms = 0.0

            qwen_metrics_list.append({
                "ndcg@10": qwen_eval["ndcg@10"],
                "mrr@10": qwen_eval["mrr@10"],
                "latency_ms": qwen_latency_ms,
            })

            if jev_eval["ndcg@10"] is not None:
                jev_metrics_list.append({
                    "ndcg@10": jev_eval["ndcg@10"],
                    "mrr@10": jev_eval["mrr@10"],
                    "latency_ms": jev_latency_ms,
                    "retained": jev_retained_count,
                })

            benchmark_records.append({
                "Query": query[:65] + "..." if len(query) > 65 else query,
                "Candidates": len(docs),
                "Qwen nDCG@10": round(qwen_eval["ndcg@10"], 3),
                "Jev nDCG@10": round(jev_eval["ndcg@10"], 3)
                if jev_eval["ndcg@10"] is not None
                else "N/A",
                "Qwen Latency (ms)": round(qwen_latency_ms, 1),
                "Jev Latency (ms)": round(jev_latency_ms, 1)
                if jev_client
                else "N/A",
                "Jev Retained Docs": jev_retained_count if jev_client else "N/A",
            })

    return (
        benchmark_records,
        cutoff_threshold,
        doc_ids,
        doc_texts,
        docs,
        effective_key,
        has_run,
        jev_client,
        jev_eval,
        jev_latency_ms,
        jev_metrics_list,
        jev_ranked_ids,
        jev_retained_count,
        jev_t0,
        query,
        query_doc_pairs,
        qwen_eval,
        qwen_latency_ms,
        qwen_metrics_list,
        qwen_model,
        qwen_order,
        qwen_ranked_ids,
        qwen_scores,
        qwen_t0,
        qrels,
        response,
        sample,
        text_to_id,
    )


@app.cell
def __(
    alt,
    benchmark_records,
    has_run,
    jev_metrics_list,
    mo,
    np,
    pd,
    qwen_metrics_list,
):
    if not has_run:
        dashboard_view = mo.md(
            "👉 Adjust configuration parameters and click **🚀 Execute Benchmark** above to run."
        )
    else:
        # Build summary statistics
        models_data = []

        if qwen_metrics_list:
            models_data.append({
                "Model": "Qwen3-Reranker",
                "Mean nDCG@10": np.mean([m["ndcg@10"] for m in qwen_metrics_list]),
                "Mean MRR@10": np.mean([m["mrr@10"] for m in qwen_metrics_list]),
                "P50 Latency (ms)": np.median([m["latency_ms"] for m in qwen_metrics_list]),
                "P95 Latency (ms)": np.percentile([m["latency_ms"] for m in qwen_metrics_list], 95),
            })

        if jev_metrics_list:
            models_data.append({
                "Model": "Jev (TypeSafe API)",
                "Mean nDCG@10": np.mean([m["ndcg@10"] for m in jev_metrics_list]),
                "Mean MRR@10": np.mean([m["mrr@10"] for m in jev_metrics_list]),
                "P50 Latency (ms)": np.median([m["latency_ms"] for m in jev_metrics_list]),
                "P95 Latency (ms)": np.percentile([m["latency_ms"] for m in jev_metrics_list], 95),
            })

        df_summary = pd.DataFrame(models_data)

        # Chart 1: Quality comparison (nDCG@10 & MRR@10)
        df_melted_quality = df_summary.melt(
            id_vars=["Model"],
            value_vars=["Mean nDCG@10", "Mean MRR@10"],
            var_name="Metric",
            value_name="Score",
        )

        quality_chart = (
            alt.Chart(df_melted_quality)
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", title=None),
                y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", scale=alt.Scale(scheme="tableau10")),
                xOffset="Model:N",
            )
            .properties(width=280, height=220, title="Relevance Quality")
        )

        # Chart 2: Latency profile (P50 vs P95)
        df_melted_latency = df_summary.melt(
            id_vars=["Model"],
            value_vars=["P50 Latency (ms)", "P95 Latency (ms)"],
            var_name="Percentile",
            value_name="Milliseconds",
        )

        latency_chart = (
            alt.Chart(df_melted_latency)
            .mark_bar()
            .encode(
                x=alt.X("Percentile:N", title=None),
                y=alt.Y("Milliseconds:Q", title="Latency (ms)"),
                color=alt.Color("Model:N", scale=alt.Scale(scheme="tableau10")),
                xOffset="Model:N",
            )
            .properties(width=280, height=220, title="Latency Profile")
        )

        dashboard_view = mo.vstack([
            mo.md("## 📊 Evaluation Results"),
            mo.hstack([mo.ui.table(df_summary), quality_chart, latency_chart]),
            mo.md("## 🔬 Per-Query Breakdown"),
            mo.ui.table(pd.DataFrame(benchmark_records)),
        ])

    dashboard_view
    return (
        dashboard_view,
        df_melted_latency,
        df_melted_quality,
        df_summary,
        latency_chart,
        models_data,
        quality_chart,
    )


if __name__ == "__main__":
    app.run()
