import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("xiaodongwang/llava-next-H20/gepa92eb")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("llava-next-dpo-image-0130.csv")