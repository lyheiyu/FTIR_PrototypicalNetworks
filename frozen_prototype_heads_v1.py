"""Class-local frozen-feature prototypes; no encoder or gradient training."""
import numpy as np
import copy


class FrozenPrototype:
    def __init__(self, kind, dimension, prior=None, reg=1e-3, tau=5.):
        if kind not in ('euclidean', 'cosine', 'shared', 'shrinkage'):
            raise ValueError(kind)
        self.kind, self.dimension, self.reg, self.tau = kind, dimension, reg, tau
        self.centers, self.precisions, self.counts = {}, {}, {}
        self.global_matrix = None
        if kind in ('shared', 'shrinkage'):
            if prior is None or prior.shape != (dimension, dimension):
                raise ValueError('A base-training-only covariance is required')
            self.global_matrix = (np.linalg.inv(prior + reg * np.eye(dimension))
                                  if kind == 'shared' else np.array(prior, copy=True))

    def register(self, label, x):
        """Only the arriving class is computed; existing class state is untouched."""
        x = np.asarray(x, dtype=np.float64)
        if label in self.centers or x.ndim != 2 or not len(x) or x.shape[1] != self.dimension:
            raise ValueError('Register one new class with nonempty permitted support')
        mean = x.mean(axis=0)
        if self.kind == 'cosine':
            norm = np.linalg.norm(mean)
            mean = mean / norm if norm else mean
        if self.kind == 'shrinkage':
            residual = x - mean
            cov = residual.T @ residual / len(x)
            alpha = (len(x) - 1) / (len(x) - 1 + self.tau)
            self.precisions[label] = np.linalg.inv(alpha * cov + (1 - alpha) * self.global_matrix
                                                   + self.reg * np.eye(self.dimension))
        self.centers[label], self.counts[label] = mean, len(x)

    def predict(self, x):
        classes = sorted(self.centers)
        means = np.stack([self.centers[c] for c in classes])
        if self.kind == 'cosine':
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            normalized = x / np.where(norms > 0, norms, 1.)
            distance = 1 - normalized @ means.T
        else:
            diff = x[:, None, :] - means[None, :, :]
            if self.kind == 'euclidean':
                distance = np.sum(diff * diff, axis=2)
            elif self.kind == 'shared':
                distance = np.einsum('mci,ij,mcj->mc', diff, self.global_matrix, diff, optimize=True)
            else:
                distance = np.stack([np.sum((diff[:, i] @ self.precisions[c]) * diff[:, i], axis=1)
                                     for i, c in enumerate(classes)], axis=1)
        return np.asarray(classes)[np.argmin(distance, axis=1)]

    def storage(self):
        centers = sum(x.nbytes for x in self.centers.values())
        metrics = sum(x.nbytes for x in self.precisions.values())
        global_bytes = self.global_matrix.nbytes if self.global_matrix is not None else 0
        return {'stored_feature_bank_bytes': centers, 'class_metric_bytes': metrics,
                'global_metric_bytes': global_bytes, 'count_bytes': 8 * len(self.counts),
                'total_numeric_state_bytes': centers + metrics + global_bytes + 8 * len(self.counts)}

    def clone(self):
        result = copy.copy(self)
        result.centers, result.precisions, result.counts = self.centers.copy(), self.precisions.copy(), self.counts.copy()
        return result
