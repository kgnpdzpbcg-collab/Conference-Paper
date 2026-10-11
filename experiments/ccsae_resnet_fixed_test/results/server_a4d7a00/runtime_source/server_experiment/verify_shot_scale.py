"""核验冻结 P1 的 K10/40 配对；K20 仅作配置、数据池和固定 Val 参考。"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import configure, dump, file_sha, state_sha
import torch
import ccsae.fewshot_train as ft
from ccsae.model import CCSAE
from ccsae.fewshot_data import make_fewshot_split
from ccsae.prototype import prototype_logits

ROOT = Path(__file__).resolve().parents[1]


def manual_metrics(truth, predicted, classes):
    """由混淆矩阵独立复算，避免仅重复调用训练入口的 sklearn 指标函数。"""
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    tp = matrix.diagonal().astype(float)
    actual, selected = matrix.sum(1), matrix.sum(0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual>0)
    denominator = actual + selected
    f1 = np.divide(2*tp, denominator, out=np.zeros_like(tp), where=denominator>0)
    return {"accuracy":float(tp.sum()/matrix.sum()), "macro_f1":float(f1.mean()),
        "balanced_accuracy":float(recall.mean())}, matrix, f1


@torch.no_grad()
def encoded(model, values):
    """沿用上游的批大小 64，只执行编码器以独立重建分类输出。"""
    return torch.cat([model.feature_vector(model.encode(chunk.cuda())) for chunk in values.split(64)])


@torch.no_grad()
def verify_job(job):
    cfg_args = json.loads((job / "config.json").read_text())
    k = cfg_args["k"]
    assert k in (10,40) and cfg_args["val_per_class"] == 5
    assert cfg_args["support"] == 3 and cfg_args["query"] == 2
    assert cfg_args["stage1_epochs"] == 10
    assert cfg_args["epochs"] == 40
    assert cfg_args["steps_per_epoch"] == 10
    assert cfg_args["mode"] == "both" and not cfg_args["tiny"]
    assert cfg_args["metric"] == "euclidean" and cfg_args["temperature"] == 1
    assert cfg_args["distance_normalization"] == "none"
    case, seed = cfg_args["case"], cfg_args["seed"]
    phase = "seed42_precheck" if seed == 42 else "seeds43_44_followup"
    old = ROOT / "outputs/proto_scale_p1/server_a4d7a00" / phase / case / f"k20_sample{seed}_train{seed}_distnone"
    original_cfg = json.loads((old / "config.json").read_text())
    assert {k:v for k,v in cfg_args.items() if k not in ("output","k")} == {k:v for k,v in original_cfg.items() if k not in ("output","k")}
    assert cfg_args["lr"] == (.0022 if case == "skl" else .0102)
    assert cfg_args["scl_weight"] == (.0017 if case == "skl" else .0059)
    split = make_fewshot_split(ROOT.parent / "Data.zip", case, k=k, sample_seed=seed)
    saved_audit = json.loads((job / "split_audit.json").read_text())
    assert saved_audit == split.audit
    preflight = json.loads((job.parents[2] / "provenance/splits" / f"{case}_k{k}_seed{seed}.json").read_text())
    assert split.audit == preflight
    k20_audit = json.loads((old / "split_audit.json").read_text())
    for name in ("train_pool_sha256", "val_pool_sha256", "test_pool_sha256", "selected_val_hashes_by_class"):
        assert split.audit[name] == k20_audit[name]
    for c, hashes in split.audit["selected_train_hashes_by_class"].items():
        if k == 10:
            assert hashes == k20_audit["selected_train_hashes_by_class"][c][:10]
        else:
            assert hashes[:20] == k20_audit["selected_train_hashes_by_class"][c]
    shared = torch.load(job / "shared_stage1_last.pt", map_location="cpu", weights_only=True)["state_dict"]
    assert all(torch.isfinite(t).all() for t in shared.values())
    trace = json.loads((job / "pair_verification.json").read_text())
    assert trace["stage1_last_sha256"] == file_sha(job / "shared_stage1_last.pt")
    assert state_sha(shared) == trace["methods"]["fc"]["initial_state_sha256"] == trace["methods"]["proto"]["initial_state_sha256"]
    feature_history = json.loads((job / "shared_stage1_history.json").read_text())
    assert len(feature_history) == cfg_args["stage1_epochs"]
    for row in feature_history:
        assert all(np.isfinite(row[key]) for key in ("total","reconstruction","scl"))
        assert np.isclose(row["total"], row["reconstruction"]+cfg_args["scl_weight"]*row["scl"],rtol=1e-5,atol=1e-5)
    # 独立重放 episode RNG，检查保存的实际索引、类平衡和两路严格一致。
    arrays = np.load(job / "episodes.npz")
    y = split.train_y.numpy()
    rng = np.random.default_rng(seed+901)
    count = cfg_args["epochs"]*cfg_args["steps_per_epoch"]
    for part in ("support","query"):
        assert np.array_equal(arrays[f"fc_{part}"], arrays[f"proto_{part}"])
    classes = split.audit["classes"]
    for step in range(count):
        expected_s, expected_q = [],[]
        for c in range(classes):
            drawn = rng.choice(np.flatnonzero(y==c), 5, replace=False)
            expected_s.extend(drawn[:3])
            expected_q.extend(drawn[3:])
        s,q = arrays["fc_support"][step],arrays["fc_query"][step]
        assert np.array_equal(s,expected_s) and np.array_equal(q,expected_q)
        assert not set(s)&set(q)
        assert np.bincount(y[s],minlength=classes).tolist() == [3]*classes
        assert np.bincount(y[q],minlength=classes).tolist() == [2]*classes
    summary = json.loads((job / "summary.json").read_text())
    assert summary["test_evaluated"] is False
    methods = {}
    for method in ("fc","proto"):
        history = json.loads((job / f"{method}_history.json").read_text())
        assert len(history) == cfg_args["epochs"]
        assert all(np.isfinite(list(row.values())).all() for row in history)
        best = max(history,key=lambda row:row["macro_f1"])
        checkpoint = torch.load(job / f"{method}_best.pt",map_location="cpu",weights_only=True)
        assert checkpoint["epoch"] == best["epoch"] == summary["models"][method]["best_epoch"]
        assert checkpoint["distance_normalization"] == "none"
        state = checkpoint["state_dict"]
        assert all(torch.isfinite(t).all() for t in state.values())
        for name,tensor in shared.items():
            if name.startswith("decoder.") or (method=="proto" and name.startswith("classifier.")):
                assert torch.equal(tensor,state[name]), f"不应训练的权重被更新：{name}"
        model = CCSAE(ft.make_config(case,False)).cuda().eval()
        model.load_state_dict(state)
        val_features = encoded(model,split.val_x)
        numerics = None
        if method == "fc":
            scores = model.classifier(val_features)
        else:
            train_features = encoded(model,split.train_x)
            centers = torch.stack([train_features[split.train_y.cuda()==c].mean(0) for c in range(classes)])
            scores = prototype_logits(val_features,centers,distance_normalization="none")
            # 直接广播平方距离与上游 cdist 输出比较，检查维度缩放及分类方向。
            independent = -((val_features[:,None,:]-centers[None,:,:]).square().sum(-1))
            # 浮点实现遇到近并列时允许排序差异，但必须处于逐行误差上界内；不放宽重载预测一致要求。
            mismatched = scores.argmax(1) != independent.argmax(1)
            row_error = (scores-independent).abs().max(1).values
            ordered = scores.topk(2,dim=1).values
            margin = ordered[:,0]-ordered[:,1]
            assert bool((margin[mismatched] <= 2*row_error[mismatched]+1e-12).all())
            stable = -((val_features.double()[:,None,:]-centers.double()[None,:,:]).square().sum(-1))
            # 使用随特征范数和嵌入维度变化的舍入预算，避免把固定绝对容差当作所有尺度的标准。
            # gamma_D=D*eps/(1-D*eps) 是 D 项累加的保守误差项；预算涵盖范数、点积及 sqrt 后平方。
            eps = torch.finfo(torch.float32).eps
            width = val_features.shape[1]
            gamma = width*eps/(1-width*eps)
            norm_scale = (val_features.double().square().sum(1)[:,None]+centers.double().square().sum(1)[None,:])
            budget = 8*gamma*norm_scale+8*eps*stable.abs()+1e-12
            stable_error = (scores.double()-stable).abs()
            assert bool((stable_error <= budget).all())
            assert bool(((independent.double()-stable).abs() <= budget).all())
            stable_pred = stable.argmax(1).cpu().numpy()
            stable_metrics,_,_ = manual_metrics(split.val_y.numpy(),stable_pred,classes)
            numerics = {"direct_formula_prediction_differences": int(mismatched.sum()),
                "fp64_distance_prediction_differences": int((scores.argmax(1).cpu().numpy()!=stable_pred).sum()),
                "validation_rows": len(stable_pred), "max_absolute_score_difference": float(row_error.max()),
                "near_tie_bound_verified": True, "scale_aware_roundoff_bound_verified":True,
                "max_abs_fp64_score_difference":float(stable_error.max()),
                "max_error_divided_by_eps_norm_scale":float((stable_error/(eps*norm_scale+1e-30)).max()), "fixed_checkpoint_fp64_distance_metrics": stable_metrics,
                "diagnostic_only_no_reselection": True}
        predicted = scores.argmax(1).cpu().numpy()
        truth = split.val_y.numpy()
        saved = json.loads((job / f"{method}_val_predictions.json").read_text())
        assert saved["truth"] == truth.tolist() and saved["predicted"] == predicted.tolist()
        metrics,matrix,per_class_f1 = manual_metrics(truth,predicted,classes)
        for name,value in metrics.items():
            assert np.isclose(value,summary["models"][method]["val"][name],atol=1e-12,rtol=0)
            assert np.isclose(value,best[name],atol=1e-12,rtol=0)
        methods[method] = {"metrics":metrics,"best_epoch":best["epoch"],"confusion_matrix":matrix.tolist(),
            "per_class_f1":per_class_f1.tolist(),"checkpoint_sha256":file_sha(job / f"{method}_best.pt"),
            "validation_predictions_exact":True,"distance_numerics":numerics}
        np.savez_compressed(job / f"{method}_verified_val.npz",truth=truth,predicted=predicted,scores=scores.cpu().numpy())
        del model
    delta = methods["proto"]["metrics"]["macro_f1"]-methods["fc"]["metrics"]["macro_f1"]
    assert np.isclose(delta,summary["paired_val_macro_f1_delta_proto_minus_fc"],atol=1e-12)
    return {"case":case,"k":k,"seed":seed,"status":"passed","methods":methods,"paired_delta":delta, "paired_initialization_identical":True, "paired_episode_stream_identical":True, "frozen_config_matches_k20_except_k_and_output":True, "distance_normalization":"none"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--expected-pairs",type=int,default=12)
    args=parser.parse_args()
    configure()
    out=Path(args.output)
    jobs=sorted(out.glob("*/k*_sample*_train*"))
    assert len(jobs)==args.expected_pairs
    results=[verify_job(job) for job in jobs]
    dump(out/"verification.json",{"status":"passed","paired_runs":len(results),"test_evaluated":False,"results":results})
    print(f"独立核验通过：{len(results)} 组配对，初始化/实际 Episode/冻结参数/最佳权重/验证预测/指标均一致；无 Test 推理。")

if __name__ == "__main__":
    main()
