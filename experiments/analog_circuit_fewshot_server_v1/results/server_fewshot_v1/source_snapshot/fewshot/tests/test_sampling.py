"""检查少样本实验中容易导致额外数据泄漏或错误抽样的核心约束。"""

import numpy as np
from fewshot_data import normalize, select_indices


def test_nested_balanced_reproducible():
    """同种子结果完全相同，各类精确 K 条，且 K10/K20/K40 逐级包含。"""
    labels=np.repeat(np.arange(3),[70,210,210])
    first=select_indices(labels,42)
    second=select_indices(labels,42)
    for k in (10,20,40):
        assert np.array_equal(first[k],second[k])
        assert np.bincount(labels[first[k]]).tolist()==[k]*3
        assert len(np.unique(first[k]))==3*k
    assert set(first[10])<set(first[20])<set(first[40])
    assert not np.array_equal(first[40],select_indices(labels,43)[40])


def test_normalizer_ignores_validation_values():
    """验证集的数值怎样变化都不能改变训练统计或标准化后的训练输入。"""
    train=np.arange(20,dtype=np.float64).reshape(2,10)
    a,ma=normalize(train,{'train':train,'val':np.ones((1,10))})
    b,mb=normalize(train,{'train':train,'val':np.full((1,10),1e10)})
    assert ma==mb and ma['mean']==float(train.mean())
    assert np.array_equal(a['train'],b['train'])
    assert a['train'].dtype==np.float32


def test_insufficient_class_is_rejected():
    """任意类别不足最大 K 时应报错，不能重复采样伪造少样本数目。"""
    import pytest
    with pytest.raises(ValueError,match='不足 40'):
        select_indices(np.repeat(np.arange(2),[39,100]),42)
