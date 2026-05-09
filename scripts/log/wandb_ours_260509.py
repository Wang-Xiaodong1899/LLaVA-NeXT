
import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
# run = api.run("xiaodongwang/llava-next-8H20/9o9o363m")

# run = api.run("xiaodongwang/llava-next-8H20/fy7zhxmn")

# run = api.run("xiaodongwang/llava-next-8H20/naw7egw3")

# run = api.run("xiaodongwang/llava-next-8H20/0mpdywp3")

# run = api.run("xiaodongwang/llava-next-8H20/bsesdn2i")

run = api.run("xiaodongwang/llava-next-8H20/k045ihut")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
# metrics_dataframe.to_csv("lr0-ours-win-ours-rej-0514.csv")
# metrics_dataframe.to_csv("lr0-hound-win-hound-rej-0514.csv")
# metrics_dataframe.to_csv("lr0-hound-gt-text-hallu-rej-0514.csv")
# metrics_dataframe.to_csv("lr0-input-chosen-text-hallu-rej-0514.csv")
# metrics_dataframe.to_csv("lr5e-7-ours-win-ours-rej-0514.csv")
metrics_dataframe.to_csv("lr5e-7-hound-win-hound-rej-0514.csv")