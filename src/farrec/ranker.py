"""Stage 2: learning-to-rank over candidates (LightGBM lambdarank or CatBoost YetiRank).

The ranker behaves like any other recommender (same call signature), so it is
evaluated by exactly the same protocol. For an evaluation cutoff T it trains on
earlier cutoffs only:
    valid cutoff  = last cutoff c with c + target_days <= T      (early stopping)
    train cutoffs = cutoffs c with c + target_days <= valid       (no label overlap with valid)
Group for ranking = (cutoff, customer).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FarTrans
from .features import CAT_FEATURES, build_frame, feature_columns
from .splits import all_cutoffs, make_slice, train_cutoffs_for

LGBM_PARAMS = dict(objective="lambdarank", n_estimators=3000, learning_rate=0.03, num_leaves=63,
                   min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                   reg_lambda=1.0, lambdarank_truncation_level=20, verbose=-1, random_state=0)
CATBOOST_PARAMS = dict(loss_function="YetiRank", iterations=3000, learning_rate=0.05, depth=6,
                       eval_metric="NDCG:top=10", random_seed=0, verbose=0, thread_count=-1)


class TwoStageRanker:
    def __init__(self, ft: FarTrans, cfg: dict, backend: str = "lgbm", n_train_cutoffs: int = 12,
                 cache_dir: str | Path = "data/processed/frames", drop_features: tuple = (), params: dict | None = None,
                 suitable_only: bool = False):
        self.ft, self.cfg, self.backend = ft, cfg, backend
        self.n_train = n_train_cutoffs
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.drop = set(drop_features)
        # suitable_only: train and rank only among assets allowed by the default (strict) risk policy,
        # instead of training on everything and filtering afterwards
        self.suitable_only = suitable_only
        self.variant = "_suitable" if suitable_only else ""
        self.params = params or (LGBM_PARAMS if backend == "lgbm" else CATBOOST_PARAMS)
        s = cfg["split"]
        self.td = s["target_days"]
        # training cutoffs may start earlier than the evaluation range
        self.cuts = all_cutoffs("2019-01-01", s["last_cutoff"], s["freq"])
        self.models: dict = {}
        self.categories: dict = {}  # cutoff -> {feature: categories}; one model per cutoff
        self.score_dir = self.cache.parent / f"scores_{backend}{self.variant}"
        self.score_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.cache.parent / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ data
    def frame(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Labelled candidate frame for customers that bought something in the target window."""
        path = self.cache / f"{cutoff.date()}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        sl = make_slice(self.ft, cutoff, self.td)
        df = build_frame(self.ft, cutoff, sl.eval_customers, sl.candidates, self.cfg, sl.target)
        df.to_parquet(path, index=False)
        return df

    def _restrict(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.p_risk_excess.fillna(1) <= 0] if self.suitable_only else df

    def features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in feature_columns(df) if c not in self.drop]

    def _prep(self, df: pd.DataFrame, cutoff: pd.Timestamp, fit: bool = False) -> pd.DataFrame:
        X = df[self.features(df)].copy()
        for c in CAT_FEATURES:
            if c not in X:
                continue
            if self.backend == "catboost":
                X[c] = X[c].astype(str)
            else:
                if fit:
                    self.categories.setdefault(cutoff, {})[c] = pd.Index(sorted(X[c].astype(str).unique()))
                X[c] = pd.Categorical(X[c].astype(str), categories=self.categories[cutoff][c])
        return X

    @staticmethod
    def _groups(df: pd.DataFrame) -> np.ndarray:
        return df.groupby(["cutoff", "customerID"], sort=False).size().to_numpy()

    def _training_frames(self, eval_cutoff: pd.Timestamp):
        usable = train_cutoffs_for(eval_cutoff, self.cuts, self.td)
        valid_c = usable[-1]
        train_c = train_cutoffs_for(valid_c, self.cuts, self.td)[-self.n_train:]

        def load(cs):
            df = self._restrict(pd.concat([self.frame(c) for c in cs], ignore_index=True))
            # groups without any positive carry no ranking signal
            has_pos = df.groupby(["cutoff", "customerID"]).label.transform("max") > 0
            return df[has_pos].sort_values(["cutoff", "customerID"]).reset_index(drop=True)

        return load(train_c), load([valid_c]), train_c, valid_c

    # ------------------------------------------------------------ model
    def fit(self, eval_cutoff: pd.Timestamp):
        if eval_cutoff in self.models:
            return self.models[eval_cutoff]
        loaded = self._load_model(eval_cutoff)
        if loaded is not None:
            self.models[eval_cutoff] = loaded
            return loaded
        tr, va, train_c, valid_c = self._training_frames(eval_cutoff)
        Xtr, Xva = self._prep(tr, eval_cutoff, fit=True), self._prep(va, eval_cutoff)
        if self.backend == "lgbm":
            import lightgbm as lgb
            m = lgb.LGBMRanker(**self.params)
            m.fit(Xtr, tr.label, group=self._groups(tr), eval_set=[(Xva, va.label)],
                  eval_group=[self._groups(va)], eval_at=[10],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            best = m.best_iteration_
        else:
            from catboost import CatBoostRanker, Pool
            cats = [c for c in CAT_FEATURES if c in Xtr]
            gid = lambda d: d.cutoff.astype(str) + "|" + d.customerID  # noqa: E731
            ptr = Pool(Xtr, tr.label, group_id=gid(tr), cat_features=cats)
            pva = Pool(Xva, va.label, group_id=gid(va), cat_features=cats)
            m = CatBoostRanker(**self.params, early_stopping_rounds=100)
            m.fit(ptr, eval_set=pva, use_best_model=True)
            best = m.get_best_iteration()
        print(f"    [{self.backend}] eval {eval_cutoff.date()}: train {train_c[0].date()}..{train_c[-1].date()} "
              f"({len(tr)} rows), valid {valid_c.date()}, best_iter={best}", flush=True)
        self.models[eval_cutoff] = m
        self._save_model(eval_cutoff, m)
        return m

    # ------------------------------------------------------------ persistence
    # Models live next to the feature cache (data/processed/models) and are wiped with it,
    # so a model never outlives the feature code it was trained with.
    def _model_path(self, cutoff: pd.Timestamp) -> Path:
        ext = "txt" if self.backend == "lgbm" else "cbm"
        tag = "" if not self.drop else "_drop" + str(abs(hash(tuple(sorted(self.drop)))) % 10**6)
        return self.model_dir / f"{self.backend}{self.variant}{tag}_{cutoff.date()}.{ext}"

    def _save_model(self, cutoff, m):
        path = self._model_path(cutoff)
        if self.backend == "lgbm":
            m.booster_.save_model(str(path), num_iteration=m.best_iteration_)
            cats = {c: list(v) for c, v in self.categories[cutoff].items()}
            path.with_suffix(".json").write_text(json.dumps(cats, ensure_ascii=False), encoding="utf-8")
        else:
            m.save_model(str(path))

    def _load_model(self, cutoff):
        path = self._model_path(cutoff)
        if not path.exists():
            return None
        if self.backend == "lgbm":
            import lightgbm as lgb
            meta = path.with_suffix(".json")
            if not meta.exists():
                return None
            self.categories[cutoff] = {c: pd.Index(v) for c, v in json.loads(meta.read_text(encoding="utf-8")).items()}
            return lgb.Booster(model_file=str(path))
        from catboost import CatBoostRanker
        return CatBoostRanker().load_model(str(path))

    def score(self, df: pd.DataFrame, eval_cutoff: pd.Timestamp) -> np.ndarray:
        model = self.fit(eval_cutoff)
        return model.predict(self._prep(df, eval_cutoff))

    def feature_importance(self, eval_cutoff: pd.Timestamp) -> pd.Series:
        m = self.fit(eval_cutoff)
        if self.backend == "lgbm":
            booster = getattr(m, "booster_", m)  # LGBMRanker after training, Booster after loading
            imp = booster.feature_importance("gain")
            names = booster.feature_name()
        else:
            imp, names = m.get_feature_importance(type="PredictionValuesChange"), m.feature_names_
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def recommend_any(self, ft, cutoff, customers, candidates, k):
        """Like __call__, but for an arbitrary customer set / candidate universe (no labels at all).
        Used for the paper's protocol, whose customer set differs from ours."""
        path = self.cache / f"unlabelled_{cutoff.date()}_{len(customers)}_{len(candidates)}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = build_frame(self.ft, cutoff, pd.Index(customers), pd.Index(candidates), self.cfg)
            df.to_parquet(path, index=False)
        df = df.assign(score=self.score(df, cutoff))
        df = df.sort_values(["customerID", "score"], ascending=[True, False])
        df["rank"] = df.groupby("customerID").cumcount() + 1
        return df.loc[df["rank"] <= k, ["customerID", "ISIN", "score", "rank"]].reset_index(drop=True)

    # ------------------------------------------------------------ recommender interface
    def __call__(self, ft, cutoff, customers, candidates, k):
        df = self.frame(cutoff)
        df = self._restrict(df[df.customerID.isin(customers)].drop(columns="label"))  # label never used for prediction
        df = df.assign(score=self.score(df, cutoff))
        if self.score_dir is not None:
            # scored candidates are reused by post-processing experiments (risk / return re-ranking)
            df.to_parquet(self.score_dir / f"{cutoff.date()}.parquet", index=False)
        df = df.sort_values(["customerID", "score"], ascending=[True, False])
        df["rank"] = df.groupby("customerID").cumcount() + 1
        return df.loc[df["rank"] <= k, ["customerID", "ISIN", "score", "rank"]].reset_index(drop=True)
