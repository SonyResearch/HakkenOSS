
# KGE V2
uv run python scripts/run_pipeline.py train-kge --override \
"optimizer=adam \
optimizer.kwargs.lr=0.009 \
kge=complex \
kge.embedding_dim=128 \
negative_strategy.name=mean \
data_loader.kwargs.num_negatives=100 \
loss.name=MarginRankingLoss \
loss.kwargs.margin=31.24 \
experiment_tracker.experiment_name=kge-v2 \
trainer=default \
early_stopping.patience=20 \
trainer.max_epochs=100"