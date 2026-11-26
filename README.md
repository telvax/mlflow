# mlflow

An [MLflow](https://mlflow.org) client for [Telvax](https://github.com/telvax/telvax), over MLflow's REST API. Its parts:
* experiments and runs;
* params, metrics and tags;
* artifacts (a `std.model` file, for one);
* the model registry.

MLflow keeps its server, its storage and its UI.

```sh
tvpm add mlflow
```

```telvax
from std.bytes import Bytes
from mlflow.client import from_env
from mlflow.tracking import get_or_create, start_run, log_param, log_metric, log_artifact, end_run
from mlflow.registry import create_registered_model, create_model_version, set_registered_model_alias

fn track(model_file: Bytes, r2: Float) -> Result[Text, Text]:
    let c = from_env()?                                   # MLFLOW_TRACKING_URI
    let run = start_run(c, get_or_create(c, "sales")?, "forest")?
    log_param(c, run.run_id, "n_estimators", "100")?
    log_metric(c, run.run_id, "r2", r2, 0)?
    log_artifact(c, run, "model/model.safetensors", model_file)?
    end_run(c, run.run_id, "FINISHED")?
    create_registered_model(c, "sales")?
    let version = create_model_version(c, "sales", "runs:/" + run.run_id + "/model", run.run_id)?
    set_registered_model_alias(c, "sales", "champion", version)?
    Ok(version)
```

## The client (`mlflow.client`)

`from_env()` connects to `MLFLOW_TRACKING_URI`, and `connect(uri)` to any address. Both take the credentials of the
environment: `MLFLOW_TRACKING_TOKEN` (a bearer token), or `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` (basic
authentication).

An error answer of MLflow is an `Err` with its status, code and message
(`MLflow answered 404: RESOURCE_DOES_NOT_EXIST: ...`).

## Tracking (`mlflow.tracking`)

| Function | What it does |
|---|---|
| `get_or_create(c, name) -> Result[Text, Text]` | The id of an experiment, created when there is none. |
| `start_run(c, experiment_id, name) -> Result[RunInfo, Text]` | Starts a run. `RunInfo` has `run_id`, `experiment_id`, `artifact_uri` and `status`. |
| `end_run(c, run_id, status)` | Ends it: `FINISHED`, `FAILED` or `KILLED`. |
| `get_run(c, run_id) -> Result[RunInfo, Text]` | What MLflow says of a run now. |
| `log_param(c, run_id, key, value)` | A parameter. MLflow keeps the first value, so logging another value for the same key is an error. |
| `log_metric(c, run_id, key, value, step)` | A value of a metric at a step, timed now. |
| `log_batch(c, run_id, params, metrics, tags)` | Many at once, with `Param`, `Metric` and `Tag`. |
| `set_tag(c, run_id, key, value)` | A tag. |
| `log_artifact(c, run, path, data: Bytes)` | Uploads a file to the run's artifacts, through the artifact proxy of the server. |

`log_artifact` needs a server that serves artifacts, which is the default of `mlflow server` since 2.0 (`mlflow-artifacts:`
locations).

## Registry (`mlflow.registry`)

| Function | What it does |
|---|---|
| `create_registered_model(c, name)` | Registers a name. One that already exists is fine. |
| `create_model_version(c, name, source, run_id) -> Result[Text, Text]` | A new version, from `runs:/<run id>/<path>`; gives its number. |
| `set_registered_model_alias(c, name, alias, version)` | Points an alias (`champion`) at a version. |

## Tests

* `telvax test src/client_test.tvx` checks the URLs, the encoding and the error answers.
* `tests/server` logs an experiment, a run (params, metrics over steps, tags, a batch, an artifact) and a model version
  with an alias to a real MLflow server. `check.py` then reads it all back with MLflow's own Python client: params,
  metric history, artifact listing and content, version, alias. CI starts the server.

Licensed under MIT or Apache-2.0, at your choice.
