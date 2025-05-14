import pandas as pd
import wandb

api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
# run = api.run("xiaodongwang/llava-next-jf-4A100/63lt1stl")
run = api.run("xiaodongwang/llava-next-PKU-4A100/3gymyevv")

# save the metrics for the run to a csv file
history = run.scan_history()

df = pd.DataFrame(history)

df.to_csv("llava-next-ours-0304.csv")