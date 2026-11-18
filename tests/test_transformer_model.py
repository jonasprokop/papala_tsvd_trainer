import numpy as np
import torch
from torch import nn
from transformer_model import fit_masked_ridge_cv, TransformerNeuralModel
from transformer_model.ridge import fit_masked_ridge_cv_reference
from transformer_model.model import export_model, load_export

class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 6)
    def forward(self, x):
        return self.layer(x)

def _forward(backbone, image):
    return backbone(image)

def test_masked_ridge_and_export(tmp_path):
    rng = np.random.default_rng(41)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    Y = X @ np.array([[1., -2.], [0.4, 0.2], [-0.5, 1.5]]) + 0.2
    mask = np.ones_like(Y, bool)
    mask[::3, 1] = False
    fit = fit_masked_ridge_cv(X, Y, mask, train_indices=np.arange(40), groups=np.arange(50), cv_folds=4)
    pred = fit.readout(torch.from_numpy(X)).detach().numpy()
    assert pred.shape == (50,2)
    assert np.all(np.isfinite(pred))
    assert fit.valid_train_counts.tolist() == [40, 26]
    assert fit.valid_neuron_mask.tolist() == [True, True]
    assert np.corrcoef(pred[:40,0], Y[:40,0])[0,1] > .99
    bb = ToyBackbone()
    feat = bb(torch.from_numpy(X)).detach().numpy()
    fit2 = fit_masked_ridge_cv(feat, Y, mask, train_indices=np.arange(40), groups=np.arange(50), cv_folds=4)
    model = TransformerNeuralModel(bb, fit2.readout, forward_image=_forward, layer='layer')
    before = model.predict(torch.from_numpy(X[:5])).clone()
    export_model(model, tmp_path, metadata={'backbone':'toy', 'neuron_ids':['a','b']})
    model.close()
    loaded, manifest = load_export(tmp_path, build_backbone=lambda config: ToyBackbone(), forward_image=_forward)
    torch.testing.assert_close(before, loaded.predict(torch.from_numpy(X[:5])), atol=1e-6, rtol=1e-6)
    assert manifest['neuron_ids']==['a','b']
    loaded.close()

def test_no_response_leakage():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24,4))
    Y = rng.normal(size=(24,1))
    m = np.ones_like(Y, dtype=bool)
    a = fit_masked_ridge_cv(X,Y,m,train_indices=np.arange(18), groups=np.arange(24))
    Y[18:] = 1000000
    b = fit_masked_ridge_cv(X,Y,m,train_indices=np.arange(18), groups=np.arange(24))
    torch.testing.assert_close(a.readout.weight, b.readout.weight)


def test_unfitted_neuron_has_explicit_mask():
    rng=np.random.default_rng(21)
    x=rng.normal(size=(12,3))
    y=rng.normal(size=(12,2))
    mask=np.ones_like(y,dtype=bool)
    mask[:,1]=False
    mask[:2,1]=True  # Too few observations for grouped CV.
    fit=fit_masked_ridge_cv(x,y,mask,train_indices=np.arange(10),groups=np.arange(12))
    assert fit.valid_train_counts.tolist()==[10,2]
    assert fit.valid_neuron_mask.tolist()==[True,False]
    assert torch.isnan(fit.readout.bias[1])


def test_fast_ridge_matches_reference_with_shared_masks_and_grouped_images():
    rng=np.random.default_rng(11)
    x=rng.normal(size=(42,9))
    y=x @ rng.normal(size=(9,5)) + rng.normal(scale=.05,size=(42,5))
    mask=np.ones_like(y,dtype=bool)
    mask[::7,:2]=False
    mask[::5,2:4]=False
    mask[:,4]=False
    mask[:2,4]=True  # Unfittable neuron in both implementations
    tr=np.arange(34)
    # Each image has multiple presentations; no fold may split its repeats.
    groups=np.repeat(np.arange(21),2)
    options={'train_indices':tr,'groups':groups,'alphas':(.01,.1,1.,10.),
             'cv_folds':3}
    slow=fit_masked_ridge_cv_reference(x,y,mask,**options)
    fast=fit_masked_ridge_cv(x,y,mask,**options)
    np.testing.assert_array_equal(fast.valid_neuron_mask,slow.valid_neuron_mask)
    np.testing.assert_allclose(fast.alphas,slow.alphas,equal_nan=True)
    np.testing.assert_allclose(fast.cv_scores,slow.cv_scores,rtol=1e-5,atol=1e-6,equal_nan=True)
    torch.testing.assert_close(fast.readout.weight,slow.readout.weight,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(fast.readout.bias[:4],slow.readout.bias[:4],rtol=1e-4,atol=1e-4)


def test_shared_masks_use_one_decomposition_per_fold(monkeypatch):
    rng=np.random.default_rng(12)
    x=rng.normal(size=(30,6))
    y=x @ rng.normal(size=(6,7))
    mask=np.ones_like(y,dtype=bool)
    eig=np.linalg.eigh
    calls=[]
    def tracked(matrix):
        calls.append(matrix.shape)
        return eig(matrix)
    monkeypatch.setattr(np.linalg,'eigh',tracked)
    fit=fit_masked_ridge_cv(x,y,mask,train_indices=np.arange(24),
                            groups=np.arange(30),cv_folds=3)
    assert fit.valid_neuron_mask.all()
    assert len(calls)==4  # Three folds + one final refit, for all seven outputs.
