import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("xiaodongwang/llama3-8B-Instruct/r1t4pmm7")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("llama-dpo.csv")