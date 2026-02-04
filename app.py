import pickle
import difflib
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Movie Recommendation System (Item-CF)",
    page_icon="🎬",
    layout="wide",
)

MODEL_PATH = Path("movielens100k_itemcf_topk.pkl")


# ----------------------------
# Load model artifact
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path.resolve()}\n"
            "Make sure 'movielens100k_itemcf_topk.pkl' is in the repo root."
        )
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    # expected keys: topk, id2title, k
    topk: Dict[int, List[Tuple[int, float]]] = artifact["topk"]
    id2title: Dict[int, str] = artifact["id2title"]
    k = artifact.get("k", None)

    # build title -> id for search
    title2id = {}
    for mid, title in id2title.items():
        if isinstance(title, str) and title.strip():
            title2id[title] = int(mid)

    return topk, id2title, title2id, k


def normalize_title(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def fuzzy_match_title(query: str, titles: List[str], n: int = 10) -> List[str]:
    q = normalize_title(query)
    if not q:
        return []
    # difflib works better with raw strings; we compare by normalized mapping
    # create mapping normalized -> original
    norm_map = {normalize_title(t): t for t in titles}
    norms = list(norm_map.keys())
    matches = difflib.get_close_matches(q, norms, n=n, cutoff=0.25)
    return [norm_map[m] for m in matches]


def recommend_from_history(
    history: Dict[int, float],
    topk: Dict[int, List[Tuple[int, float]]],
    id2title: Dict[int, str],
    n: int = 10,
    baseline: float = 3.0
) -> List[Tuple[int, str, float]]:
    """
    history: {item_id: rating}
    Simple weighted sum over neighbors:
      score(nb) += sim(item, nb) * (rating - baseline)
      normalize by sum(|sim|)
    """
    if not history:
        return []

    scores: Dict[int, float] = {}
    weights: Dict[int, float] = {}

    for item_id, r in history.items():
        for nb_id, sim in topk.get(int(item_id), []):
            nb_id = int(nb_id)
            if nb_id in history:
                continue
            scores[nb_id] = scores.get(nb_id, 0.0) + float(sim) * (float(r) - baseline)
            weights[nb_id] = weights.get(nb_id, 0.0) + abs(float(sim))

    ranked = []
    for mid, sc in scores.items():
        w = weights.get(mid, 0.0)
        if w > 0:
            ranked.append((mid, sc / w))

    ranked.sort(key=lambda x: x[1], reverse=True)
    ranked = ranked[:n]

    return [(mid, id2title.get(mid, "Unknown"), float(score)) for mid, score in ranked]


# ----------------------------
# UI
# ----------------------------
st.title("🎬 Movie Recommendation System")
st.caption("Lightweight Item-Based Collaborative Filtering (Top-K cosine neighbors)")

with st.spinner("Loading model artifact..."):
    topk, id2title, title2id, k = load_artifact(MODEL_PATH)

st.success(f"Loaded model ✅  (items: {len(id2title):,} | stored neighbors: top-{k if k else 'K'} per item)")

titles = list(title2id.keys())

st.markdown("### 1) Build your watch/rating history")
st.write("Search and add movies with ratings (1–5). Then generate recommendations.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    query = st.text_input("Search movie title", placeholder="e.g., Toy Story (1995)")
    suggestions = fuzzy_match_title(query, titles, n=10) if query else []
    selected_title = st.selectbox("Suggestions", options=suggestions, index=0 if suggestions else None)

    rating = st.slider("Your rating", min_value=1, max_value=5, value=4, step=1)
    add_btn = st.button("➕ Add to history", use_container_width=True)

with col_right:
    if "history" not in st.session_state:
        st.session_state.history = {}  # item_id -> rating

    if add_btn and selected_title:
        mid = title2id[selected_title]
        st.session_state.history[int(mid)] = float(rating)

    # show history
    if st.session_state.history:
        hist_rows = []
        for mid, r in st.session_state.history.items():
            hist_rows.append({"item_id": mid, "title": id2title.get(mid, "Unknown"), "rating": r})
        hist_df = pd.DataFrame(hist_rows).sort_values(["rating", "title"], ascending=[False, True])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.history = {}
                st.rerun()
        with c2:
            remove_title = st.selectbox(
                "Remove one",
                options=["(select)"] + [id2title.get(mid, str(mid)) for mid in st.session_state.history.keys()]
            )
            if st.button("Remove selected", use_container_width=True):
                if remove_title != "(select)":
                    # find key
                    to_remove = None
                    for mid in list(st.session_state.history.keys()):
                        if id2title.get(mid, str(mid)) == remove_title:
                            to_remove = mid
                            break
                    if to_remove is not None:
                        st.session_state.history.pop(to_remove, None)
                        st.rerun()
    else:
        st.info("History is empty. Add at least 1–3 movies to get recommendations.")

st.markdown("---")
st.markdown("### 2) Generate recommendations")

col_a, col_b, col_c = st.columns(3)
with col_a:
    n_recs = st.number_input("Number of recommendations", min_value=5, max_value=50, value=10, step=1)
with col_b:
    baseline = st.number_input("Baseline rating (center)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
with col_c:
    gen_btn = st.button("✨ Recommend", type="primary", use_container_width=True)

if gen_btn:
    recs = recommend_from_history(
        history=st.session_state.history,
        topk=topk,
        id2title=id2title,
        n=int(n_recs),
        baseline=float(baseline)
    )

    if not recs:
        st.warning("No recommendations could be generated. Try adding more movies or different titles.")
    else:
        rec_df = pd.DataFrame(recs, columns=["item_id", "title", "score"])
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        st.caption("Score = weighted similarity sum based on your rated items (normalized). Higher is better.")
