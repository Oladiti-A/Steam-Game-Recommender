# Model Card: Steam Game Recommender (Implicit ALS)

**Author:** Oladiti Abdulahi · **Version:** 1.0 (September 2026) · **Notebook:** `steam_game_recommender_als.ipynb`

## Model details

| | |
|---|---|
| **Task** | Recommend 10 games a user has not played yet (top-K recommendation) |
| **Model** | Alternating Least Squares (ALS) with implicit feedback, Spark MLlib |
| **Selected configuration** | `rank = 10`, `regParam = 0.01`, `alpha = 1.0`, `maxIter = 15`, non-negative factors |
| **Baseline** | Most-popular games: recommend the most-played games the user has not played |
| **Tracking** | Six configurations varying `rank`, `regParam` and `alpha`, each logged with MLflow |
| **Framework** | Python, PySpark, MLlib, MLflow |

## Intended use

- **Intended:** suggesting games to existing players on a games platform, as one input to a recommendations page alongside editorial and popular lists.
- **Out of scope:** brand-new users or games with no play history (cold start); decisions with financial or personal consequences; any use that would push excessive play.

## Training and evaluation data

- **Source:** the public Steam Video Games dataset (steam-200k), about 200,000 user–game interactions.
- **Preference signal:** **hours played**, not purchases, because many purchased games are never launched, so a purchase cannot distinguish a favourite game from an unplayed one.
- **Pre-processing:** exact duplicates removed; play hours summed per user–game pair; `log1p` transform applied to reduce the influence of extremely heavy players without deleting data; integer game IDs created with `StringIndexer`.
- **Split:** random 80/20 split of user–game interactions. Only users present in both training and test data are evaluated.

## Metrics

Implicit ALS predicts **preference scores, not hours**, so RMSE against hours played is not meaningful. The model is evaluated as a recommender:

- **Precision@10:** share of the 10 recommendations that the user actually played in the test set
- **Recall@10:** share of the user's test-set games that appeared in their top 10
- **NDCG@10:** rewards placing relevant games near the top of the list

Games already played in training are excluded before choosing the top 10.

## Results (test users)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| **ALS (rank 10, reg 0.01, alpha 1)** | **0.070** | **0.253** | **0.189** |
| Popularity baseline | 0.064 | 0.248 | 0.188 |

- ALS improves Precision@10 by about **9%** and Recall@10 by about 2%; NDCG@10 is effectively tied.
- Increasing `alpha` or `rank` consistently lowered all metrics, suggesting larger models overfit this sparse data.

## Limitations

- **Modest gain:** the popularity baseline is strong because a few titles dominate play activity. Differences between the top configurations were not tested for statistical significance.
- **Popularity bias:** recommendations lean towards widely played games; an example strategy-game player received mostly popular action titles.
- **Cold start:** new users and games cannot be recommended.
- **No timestamps:** a random split is used, so the evaluation does not simulate predicting future behaviour from past behaviour.
- **Hours as a proxy:** long play time does not always mean enjoyment (e.g. idling or grinding).

## Ethical considerations

- The dataset contains anonymised user IDs only; no personal information is used.
- Recommenders can reinforce popularity and reduce exposure for smaller or independent games. Diversity checks are advised before deployment.
- Optimising purely for play time could encourage excessive gaming; engagement metrics should be balanced with user wellbeing.

## Recommendations and next steps

- Combine with content features (genres, tags) to handle cold start.
- Re-rank for diversity and novelty to reduce popularity bias.
- Use a time-based split and cross-validation, with significance testing between configurations.
