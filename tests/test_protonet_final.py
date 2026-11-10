import ast
from pathlib import Path
import numpy as np
import tensorflow as tf
from protonet_training_final import prototype_loss, episode_indices
from encoder_architecture_models import build_encoder
from frozen_prototype_heads_v1 import FrozenPrototype
from run_full_mahalanobis_ablation_v1 import FullClassMahalanobis


def old_functions(filename, names):
    tree = ast.parse(Path(filename).read_text(encoding='utf-8-sig'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in selected: n.decorator_list = []
    scope = {'np': np, 'tf': tf}
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), scope)
    return scope


def test_objective_and_support_gradient():
    rng = np.random.default_rng(10)
    s = tf.Variable(rng.normal(size=(20, 3)).astype('float32'))
    q = tf.Variable(rng.normal(size=(40, 3)).astype('float32'))
    sy, qy = tf.repeat(tf.range(4), 5), tf.repeat(tf.range(4), 10)
    with tf.GradientTape() as tape: loss, _ = prototype_loss(s, q, sy, qy)
    grads = tape.gradient(loss, [s, q])
    assert all(g is not None and np.isfinite(g.numpy()).all() and np.linalg.norm(g.numpy()) > 0 for g in grads)
    old = old_functions('FTIR_new_protonetwork.py', ['build_prototypes', 'prototypical_loss_and_acc'])
    identity = lambda x, training: x
    legacy_loss, _ = old['prototypical_loss_and_acc'](identity, s, sy, q, qy)
    np.testing.assert_allclose(loss.numpy(), legacy_loss.numpy(), rtol=1e-6)
    # A numerical derivative detects accidental stop_gradient through support means.
    eps = .002; plus=s.numpy(); minus=s.numpy(); plus[0,0]+=eps; minus[0,0]-=eps
    fd=(prototype_loss(plus,q,sy,qy)[0]-prototype_loss(minus,q,sy,qy)[0])/(2*eps)
    np.testing.assert_allclose(fd.numpy(), grads[0].numpy()[0,0], atol=2e-4)


def test_m1_gradients_and_sampler():
    tf.keras.utils.set_random_seed(0); model=build_encoder('M1')
    assert model.count_params()==47400 and model.output_shape[-1]==128
    x=tf.random.normal((60,1868)); sy,qy=tf.repeat(tf.range(4),5),tf.repeat(tf.range(4),10)
    with tf.GradientTape() as tape:
        z=model(x,training=True); loss,_=prototype_loss(z[:20],z[20:],sy,qy)
    grads=tape.gradient(loss,model.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)
    assert sum(np.linalg.norm(g.numpy()) for g in grads)>0
    labels=np.repeat(['a','b','c','d'],20)
    s,q=episode_indices(labels,np.arange(80),['d','b','a','c'],np.random.default_rng(0))
    assert not set(s)&set(q)
    np.testing.assert_array_equal(labels[s],np.repeat(['d','b','a','c'],5))
    np.testing.assert_array_equal(labels[q],np.repeat(['d','b','a','c'],10))


def test_full_and_shrinkage_numerically():
    rng=np.random.default_rng(2); x=rng.normal(size=(5,4)); y=rng.normal(size=(7,4)); q=rng.normal(size=(9,4))
    prior=np.eye(4)*.3+np.ones((4,4))*.1
    for kind in ['full','shrinkage']:
        head=FullClassMahalanobis(4,reg=.001) if kind=='full' else FrozenPrototype('shrinkage',4,prior,reg=.001,tau=5.)
        scores=[]
        for c,z in [('a',x),('b',y)]:
            head.register(c,z); r=z-z.mean(0); cov=r.T@r/len(z)
            if kind=='shrinkage':
                alpha=(len(z)-1)/(len(z)-1+5); cov=alpha*cov+(1-alpha)*prior
            direct=np.array([d@np.linalg.solve(cov+.001*np.eye(4),d) for d in q-z.mean(0)])
            matrix=np.sum(((q-z.mean(0))@head.precisions[c])*(q-z.mean(0)),axis=1)
            np.testing.assert_allclose(direct,matrix,rtol=1e-12);scores.append(direct)
        np.testing.assert_array_equal(head.predict(q),np.array(['a','b'])[np.argmin(scores,axis=0)])
    legacy=old_functions('FTIR_ProtoNetwork4.py',['maha_distance_full','dists_maha_full'])
    means=np.stack([x.mean(0),y.mean(0)]); precision=np.stack([head.precisions[c] for c in ['a','b']])
    reference=np.stack([np.einsum('ni,ij,nj->n',q-m,p,q-m) for m,p in zip(means,precision)],axis=1)
    np.testing.assert_allclose(legacy['maha_distance_full'](q,means,precision).numpy(),reference,rtol=1e-12)
    assert not np.allclose(legacy['dists_maha_full'](q,means,precision).numpy(),reference)
