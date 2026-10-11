"""重载六组 Stage1，核验 D0 的模型状态恢复、确定性复算与旧输入指纹。

独立用 sklearn 重算固定原型的验证指标；固定同一数据和参数，不选择新模型。
返回验证 JSON，不修改原诊断结果、旧权重或任何 Test 输出。
"""
from pathlib import Path
import json
import math
import torch
from sklearn.metrics import f1_score,accuracy_score
from common import configure, dump, file_sha, state_sha
from ccsae import proto_diagnostics as diag
from ccsae.fewshot_data import make_fewshot_split
from ccsae.fewshot_train import _forward_features,make_config,set_seed
from ccsae.model import CCSAE

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT.parent
OUT=ROOT/'outputs/proto_diagnostics/server_a4d7a00'
configure()
s=json.loads((OUT/'SUMMARY.json').read_text())
checks=[]
for job in s['jobs']:
 r=job['result']
 case,seed=job['case'],job['seed']
 old=Path(r['source_run_directory'])
 c=json.loads((old/'config.json').read_text())
 set_seed(seed)
 split=make_fewshot_split(WORK/'Data.zip',case,k=20,val_per_class=5,split_seed=c['split_seed'],sample_seed=c['sample_seed'],val_seed=c['val_seed'])
 assert len(split.train_y)==r['num_train'] and len(split.val_y)==r['num_validation']
 m=CCSAE(make_config(case,False)).cuda()
 m.load_state_dict(torch.load(old/'shared_stage1_last.pt',map_location='cuda',weights_only=True)['state_dict'])
 before=state_sha(m.state_dict())
 again=diag.diagnose_model(m,split,torch.device('cuda'),seed=seed,support=3,query=2,repeats=32)
 assert state_sha(m.state_dict())==before,'诊断未恢复全部参数与 BN buffer'
 assert all(again[k]==r[k] for k in again),'重复只读诊断不一致'
 m.eval()
 with torch.no_grad():
  zt,_=_forward_features(m,split.train_x,torch.device('cuda'))
  zv,_=_forward_features(m,split.val_x,torch.device('cuda'))
  centers=torch.stack([zt[split.train_y.cuda()==k].double().mean(0) for k in range(m.cfg.classes)])
  # 直接相减求 FP64 平方距离，独立于上游 float32 cdist 公式。
  distance=((zv.double()[:,None]-centers[None])**2).sum(-1)
  pred=distance.argmin(1).cpu().tolist()
  independent_f1=f1_score(split.val_y.tolist(),pred,labels=list(range(m.cfg.classes)),average='macro',zero_division=0)
  independent_acc=accuracy_score(split.val_y.tolist(),pred)
 assert math.isclose(independent_f1,r['stage1_full_K_val_macro_f1'],abs_tol=1e-12)
 assert math.isclose(independent_acc,r['stage1_full_K_val_accuracy'],abs_tol=1e-12)
 assert r['fixed_embedding_logits']['dim_vs_none_prediction_differences']==0
 assert r['fixed_embedding_logits']['dim_vs_fp64_reference_prediction_differences']==0
 for norm in ('dim','none'):
  g=r['one_episode_train_mode_gradients'][norm]
  assert all(math.isfinite(g[name]) and g[name]>0 for name in ('projection_gradient_l2','first_complex_conv_gradient_l2','episode_ce'))
 checks.append({'case':case,'seed':seed,'model_state_restored':True,'repeated_diagnostic_identical':True,
                'fp64_direct_distance_and_sklearn_metrics_match':True,'checkpoint_state_sha256':before})
 print('verified D0',case,seed,flush=True)
 del m
before=json.loads((OUT/'input_before.json').read_text())
assert all(file_sha(WORK/name)==h for name,h in before.items())
dump(OUT/'VERIFICATION.json',{'status':'passed','jobs':checks,'all_old_input_file_hashes_match':True,'test_evaluated':False,
                            'limitation':'Stage1 初始化状态的机制诊断，不是训练结束 Proto best 的表征因果分析。'})
