import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("xiaodongwang/llava-next-jf-4A100-visu/p1fwhv28")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("simpo_dpo_16k_logp_visulization.csv")