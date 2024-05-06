# %% 列出指标
from datasets import list_metrics

metrics_list = list_metrics()
print(f"{len(metrics_list)=}")
print(metrics_list[:10])

from evaluate import list_evaluation_modules

metrics_list = list_evaluation_modules()
print(f"{len(metrics_list)=}")
print(metrics_list[:10])

# %% 加载指标
from datasets import load_metric

metric = load_metric(path='glue', config_name='mrpc')
print(metric)

from evaluate import load

metric = load(path='glue', config_name='mrpc')
print(metric)
# %% 打印说明
print(metric.inputs_description)
# %% 计算指标
predictions = [0,1,0]
references  = [0,1,1]
metric.compute(predictions=predictions, references=references)
# %%
