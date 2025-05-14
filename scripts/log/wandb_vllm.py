import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
# run = api.run("xiaodongwang/llava-next-jf-4A100/63lt1stl")
run = api.run("xiaodongwang/llava-next-PKU-4A100/vhb5uqxy")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("llava-next-dpo-0228.csv")