# 方案 B：kurumi 账户下重建并运行实验

> 目标目录（服务器）：`/data1/kurumi/opt`
> 环境：`/data1/kurumi/.conda/envs/ai_exp`（已有 torch 2.5.1+cu121）
> 代码来源：`优化实验`（model_optimized.py / run_optimized.py）+ `林健忠代码\code\chapter-4`（其余 6 个文件）

---

## 一、把本文件夹传到服务器

1. 用**文件传输**把本文件夹（`opt`）从本机传到笔记本，例如放到 `D:\15 知识抽取\opt`
2. 在**笔记本 PowerShell** 上传到服务器：
```powershell
scp -r "D:\15 知识抽取\opt" kurumi@10.20.6.130:/data1/kurumi/
```
上传后服务器上就是 `/data1/kurumi/opt`

## 二、服务器上执行安装（笔记本 PowerShell）

```powershell
ssh kurumi@10.20.6.130 "bash /data1/kurumi/opt/kurumi_setup.sh"
```
它会：装依赖 → 下 DocRED → 解压 → 生成 rel2id → 下 bert-base-cased → 自检。
出现 `SETUP DONE` 即成功。

## 三、环境自检

```powershell
ssh kurumi@10.20.6.130 "bash /data1/kurumi/opt/run_v3_experiments.sh check"
```
输出应全 `OK`、`cuda True`、`ngpus 3`。

## 四、开跑 P0 主实验（3 卡并行）

```powershell
ssh kurumi@10.20.6.130 "cd /data1/kurumi/opt && nohup bash run_v3_experiments.sh e1e2 > run_e1e2.out 2>&1 &"
```

查看进度：
```powershell
ssh kurumi@10.20.6.130 "tail -n 5 /data1/kurumi/opt/run_e1e2.out"
```
汇总（关键决策点）：
```powershell
ssh kurumi@10.20.6.130 "cd /data1/kurumi/opt && /data1/kurumi/.conda/envs/ai_exp/bin/python collect_v3.py"
```

## 五、后续实验

```powershell
ssh kurumi@10.20.6.130 "cd /data1/kurumi/opt && nohup bash run_v3_experiments.sh dl_large > run_dl.out 2>&1 &"   # 下 roberta-large
ssh kurumi@10.20.6.130 "cd /data1/kurumi/opt && nohup bash run_v3_experiments.sh e4 > run_e4.out 2>&1 &"          # 编码器对照
ssh kurumi@10.20.6.130 "cd /data1/kurumi/opt && nohup bash run_v3_experiments.sh e7 > run_e7.out 2>&1 &"          # 超参
```

---

## 本包已做的补丁 / 说明

| 文件 | 处理 |
|---|---|
| `prepro.py` | 第 370 行 `save_graphs(...,"your_path")` 会因目录不存在而崩溃 → 已改为**仅当 `graph_dump/` 存在时才保存**（setup 已建该目录） |
| `rel2id.json` | 官方 HF 数据不含它 → 用 `build_rel2id.py` 从 `rel_info.json` 生成 `{Na:0, Pxxxx:1..}`，`num_class=97` |
| `run_v3_experiments.sh` | 路径改为 `/data1/kurumi/opt`；venv 用 `ai_exp`；**改用 3 卡**（GPU0/1/2） |
| `collect_v3.py` | 日志目录改为 `/data1/kurumi/opt/logs_v3` |

## 已知风险（跑起来才能确认）

1. `chapter-4` 的 `prepro/evaluation` 与 `model_optimized` 的接口**大概率兼容**（已核对 `collate_fn` 10 元组、`forward` 参数、`read_docred` 签名一致），但 96/97 维是否完全对齐需实跑验证。
2. `utils.set_seed` 里调用了 `torch.use_deterministic_algorithms(True)`，某些 CUDA 算子可能报 "does not have a deterministic implementation"。若报错，把该行注释掉即可（会略微影响严格复现）。
3. `transformers==4.30.0` 需要 `numpy<2`，setup 已把 numpy 降到 1.26.4。
4. 若 DocRED 下载失败（网络），需手动下载放到 `dataset/docred/` 和 `dataset/meta/`。

## 报错时回传
把对应日志末尾 50 行发我：
```powershell
ssh kurumi@10.20.6.130 "tail -n 50 /data1/kurumi/opt/logs_v3/<某个>.log"
```
