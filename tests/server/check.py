# Reads back, with MLflow's own Python client, what src/main.tvx logged, and checks it. Usage: check.py "<output of main>"
import re, sys
from mlflow import MlflowClient
out = sys.argv[1]
f = dict(re.findall(r"(\w+)=(\S+)", out))
c = MlflowClient()
run = c.get_run(f["run"])
assert run.info.experiment_id == f["experiment"] and f["same"] == "True", out
assert c.get_experiment(f["experiment"]).name == "telvax check"
assert run.info.status == "FINISHED" == f["status"], run.info.status
assert run.data.params == {"n_estimators": "100", "seed": "42", "max_depth": "none"}, run.data.params
assert run.data.tags["data"] == "sales.csv" and run.data.tags["lang"] == "telvax"
assert run.info.run_name == "sales forest"
hist = sorted((m.step, m.value) for m in c.get_metric_history(f["run"], "loss"))
assert hist == [(0, 1.0), (1, 0.5), (2, 1.0 / 3.0)], hist
assert run.data.metrics["r2"] == 0.81
arts = [a.path for a in c.list_artifacts(f["run"], "model")]
assert arts == ["model/model.safetensors"], arts
path = c.download_artifacts(f["run"], "model/model.safetensors")
assert open(path, "rb").read() == b"weights of the model"
mv = c.get_model_version_by_alias("sales", "champion")
assert mv.version == f["version"] and mv.run_id == f["run"] and mv.source == f"runs:/{f['run']}/model", mv
assert f["duplicate"] == "True"
assert "RESOURCE_DOES_NOT_EXIST" in out or "INVALID_PARAMETER_VALUE" in out, out
print("the server holds what the Telvax client logged")
