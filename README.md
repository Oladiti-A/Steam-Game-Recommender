# Steam Game Recommender: Collaborative Filtering with PySpark ALS

A game recommender system built on 200,000 Steam user interactions using Spark MLlib's Alternating Least Squares (ALS) with implicit feedback. Experiments are tracked with MLflow, and models are evaluated with ranking metrics against a most-popular baseline.

**Stack:** Python · PySpark · Spark SQL · MLlib (ALS) · MLflow · pandas · matplotlib

---

## Approach

- **Preference signal:** play hours were used instead of purchases. Many purchased games are never launched, so purchases alone can't tell a favourite game from an unplayed one.
- **Pre-processing:** removed duplicate records, aggregated play time per user–game pair, and applied a `log1p` transform to reduce the influence of extremely heavy players without discarding data.
- **Model:** implicit-feedback ALS, tuned over `rank`, `regParam` and `alpha`, with every run logged to MLflow.
- **Evaluation:** Precision@10, Recall@10 and NDCG@10, measuring whether each user's held-out games appear in their top 10 recommendations. RMSE is not used because implicit ALS predicts preference scores, not hours.
- **Baseline:** recommending the most-played games, which any personalised model should beat.

## Results

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| **ALS (rank 10, reg 0.01, alpha 1)** | **0.070** | **0.253** | **0.189** |
| Popularity baseline | 0.064 | 0.248 | 0.188 |

- The best ALS model improved **Precision@10 by about 9%** over the popularity baseline. NDCG@10 was effectively tied.
- **Simpler models performed best:** increasing `alpha` or `rank` lowered all metrics, suggesting larger models overfit this sparse dataset.
- **Popularity bias:** a few titles dominate play activity, which makes the baseline strong and pushes recommendations towards widely played games.

## Limitations and next steps

- **Cold start:** new users and games can't be recommended. A content-based layer (genres, tags) could cover them.
- **Diversity:** re-ranking for novelty and diversity could reduce popularity bias.
- **Tuning:** a wider hyperparameter search with cross-validation, plus significance testing between runs.
- **Evaluation split:** a time-based split would reflect real use better, but the dataset has no timestamps.

## How to run

1. Download `steam-200k.csv` from the [Steam Video Games dataset on Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games).
2. Open `steam_game_recommender_als.ipynb` in [Google Colab](https://colab.research.google.com/) and upload the CSV to the session.
3. Choose **Runtime → Run all**. The first cell installs PySpark and MLflow.

## Author

**Oladiti Abdulahi**, MSc Data Science, University of Salford
[LinkedIn](https://www.linkedin.com/in/oladiti-abdulahi-6925311a9) · [GitHub](https://github.com/Oladiti-A)
