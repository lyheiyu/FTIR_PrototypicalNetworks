import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from typing import Union as U, Tuple as T
from utils import utils


def emsc(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    order: int = 2,
    reference: np.ndarray = None,
    constituents: np.ndarray = None,
    return_coefs: bool = False
) -> U[np.ndarray, T[np.ndarray, np.ndarray]]:
    """
    EMSC preprocessing with optional constituents (e.g. PCA components)

    spectra: [n_samples, n_channels]
    wavenumbers: [n_channels]
    reference: [n_channels]
    constituents: [n_constituents, n_channels]
    """
    if reference is None:
        reference = np.mean(spectra, axis=0)

    reference = reference[:, np.newaxis]

    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]

    half_rng = rng / 2
    normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

    polynomial_columns = [np.ones(len(wavenumbers))]
    for j in range(1, order + 1):
        polynomial_columns.append(normalized_wns ** j)
    polynomial_columns = np.stack(polynomial_columns).T

    if constituents is None:
        columns = (reference, polynomial_columns)
    else:
        columns = (reference, constituents.T, polynomial_columns)

    X = np.concatenate(columns, axis=1)
    A = np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T)

    spectra_columns = spectra.T
    coefs = np.dot(A, spectra_columns)
    residues = spectra_columns - np.dot(X, coefs)

    # first coefficient is the multiplicative scaling coefficient b
    eps = 1e-12
    preprocessed_spectra = (reference + residues / (coefs[0] + eps)).T

    if return_coefs:
        return preprocessed_spectra, coefs.T

    return preprocessed_spectra


def build_class_pca_constituents(class_spectra: np.ndarray, n_components: int = 2):
    """
    Build PCA constituents for one class.
    Returns:
        reference: [n_channels]
        constituents: [n_components, n_channels] or None
    """
    reference = np.mean(class_spectra, axis=0)

    n_samples = class_spectra.shape[0]
    if n_samples < 3:
        # too few samples, skip PCA constituents
        return reference, None

    max_components = min(n_components, n_samples - 1, class_spectra.shape[1])
    if max_components < 1:
        return reference, None

    centered = class_spectra - reference
    pca = PCA(n_components=max_components)
    pca.fit(centered)
    constituents = pca.components_

    return reference, constituents


class EMSA:
    """
    Extended Multiplicative Signal Augmentation with optional PCA constituents
    """

    def __init__(
        self,
        std_of_params,
        wavenumbers,
        reference,
        constituents=None,
        order=2,
        alpha=0.3,
        constituent_scale=0.3,
        polynomial_scale=0.5,
        reference_scale=0.5
    ):
        self.order = order
        self.std_of_params = std_of_params
        self.ref = reference
        self.constituents = constituents
        self.alpha = alpha
        self.constituent_scale = constituent_scale
        self.polynomial_scale = polynomial_scale
        self.reference_scale = reference_scale
        self.X = None
        self.A = None
        self.n_constituents = 0 if constituents is None else constituents.shape[0]
        self.__create_x_and_a(wavenumbers)

    def generator(self, spectra, labels, equalize_subsampling=False, shuffle=True, batch_size=32):
        spectra = np.asarray(spectra)
        labels = np.asarray(labels)

        if self.std_of_params is None:
            coefs = np.dot(self.A, spectra.T)
            self.std_of_params = coefs.std(axis=1)

        if equalize_subsampling:
            indexes = self.__rearrange_spectra(labels)
        else:
            indexes = np.arange(len(spectra))

        cur = 0
        while True:
            if shuffle:
                si = indexes[np.random.randint(len(indexes), size=batch_size)]
            else:
                si = indexes.take(range(cur, cur + batch_size), mode='wrap')
                cur += batch_size

            yield self.__batch_transform(spectra[si]), labels[si]

    def __rearrange_spectra(self, labels):
        classes = np.unique(labels, axis=0)

        if len(labels.shape) == 2:
            grouped = [np.where(np.all(labels == l, axis=1))[0] for l in classes]
        else:
            grouped = [np.where(labels == l)[0] for l in classes]

        iters_cnt = max([len(g) for g in grouped])

        indexes = []
        for i in range(iters_cnt):
            for g in grouped:
                indexes.append(np.take(g, i, mode='wrap'))

        return np.array(indexes)

    def __create_x_and_a(self, wavenumbers):
        if wavenumbers[0] > wavenumbers[-1]:
            rng = wavenumbers[0] - wavenumbers[-1]
        else:
            rng = wavenumbers[-1] - wavenumbers[0]

        half_rng = rng / 2
        normalized_wns = (wavenumbers - np.mean(wavenumbers)) / half_rng

        polynomial_columns = [np.ones_like(wavenumbers)]
        for j in range(1, self.order + 1):
            polynomial_columns.append(normalized_wns ** j)

        basis_list = [self.ref]

        if self.constituents is not None:
            for pc in self.constituents:
                basis_list.append(pc)

        basis_list.extend(polynomial_columns)

        self.X = np.stack(basis_list, axis=1)
        self.A = np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T)

    def __batch_transform(self, spectra):
        spectra_columns = spectra.T

        coefs = np.dot(self.A, spectra_columns)
        residues = spectra_columns - np.dot(self.X, coefs)

        new_coefs = coefs.copy()

        n_params = len(coefs)
        scales = np.ones(n_params)

        # coefficient layout:
        # [reference_coeff, pc1, pc2, ..., a0, a1, a2, ...]
        scales[0] = self.reference_scale

        if self.n_constituents > 0:
            scales[1:1 + self.n_constituents] = self.constituent_scale

        poly_start = 1 + self.n_constituents
        scales[poly_start:] = self.polynomial_scale

        for i in range(n_params):
            new_coefs[i] += np.random.normal(
                0,
                self.alpha * scales[i] * self.std_of_params[i],
                len(spectra)
            )

        # keep multiplicative coefficient positive
        mask = new_coefs[0] <= 0
        if np.any(mask):
            n_resamples = mask.sum()
            low = 1e-6
            high = np.maximum(coefs[0][mask], low + 1e-6)
            new_coefs[0][mask] = np.random.uniform(low, high, n_resamples)

        eps = 1e-12
        augmented = (np.dot(self.X, new_coefs) + residues * new_coefs[0] / (coefs[0] + eps)).T
        return augmented


if __name__ == '__main__':

    # firstData, secondData, thirdData, pid1, pid2, pid3, pname1, pname2, pname3, wavenumber = get_data()
    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData11(
        'dataset/D4_4_publication11.csv', 2, 1763
    )

    waveLength = np.array(waveLength, dtype=np.float64)
    intensity = np.array(intensity, dtype=np.float64)
    polymerID = np.array(polymerID)

    PN = utils.getPN('dataset/D4_4_publication11.csv')
    n_classes = len(PN)

    cmTotal = np.zeros((n_classes, n_classes))
    scoreTotal = np.zeros(5)
    m = 0

    # ===== hyperparameters you can tune =====
    emsc_order = 2
    pca_components = 4
    alpha = 0.4
    batches_to_use = 4
    batch_size = 50
    use_pca_constituents = True
    # ========================================

    for seedNum in range(20):
        x_train, x_test, y_train, y_test = train_test_split(
            intensity, polymerID, test_size=0.7, random_state=seedNum, stratify=polymerID
        )

        augmented_x_list = []
        augmented_y_list = []

        class_original_for_plot = None
        class_emsc_for_plot = None
        class_aug_for_plot = None

        for itr in range(n_classes):
            indicesPS = np.where(y_train == itr)[0]
            class_spectra = x_train[indicesPS]

            if len(class_spectra) < 2:
                continue

            if use_pca_constituents:
                reference, constituents = build_class_pca_constituents(
                    class_spectra,
                    n_components=pca_components
                )
            else:
                reference = class_spectra.mean(axis=0)
                constituents = None

            emsc_spectra, coefs_ = emsc(
                class_spectra,
                waveLength,
                reference=reference,
                constituents=constituents,
                order=emsc_order,
                return_coefs=True
            )

            coefs_std = coefs_.std(axis=0)

            label = y_train[indicesPS]

            emsa = EMSA(
                std_of_params=coefs_std,
                wavenumbers=waveLength,
                reference=reference,
                constituents=constituents,
                order=emsc_order,
                alpha=alpha,
                constituent_scale=0.3,
                polynomial_scale=0.5,
                reference_scale=0.5
            )

            generator = emsa.generator(
                class_spectra,
                label,
                equalize_subsampling=False,
                shuffle=True,
                batch_size=batch_size
            )

            class_augmented_batches = []

            for i, batch in enumerate(generator):
                if i >= batches_to_use:
                    break

                batch_x, batch_y = batch
                class_augmented_batches.append(batch_x)

            if len(class_augmented_batches) == 0:
                continue

            class_augmented = np.concatenate(class_augmented_batches, axis=0)

            # optional filtering: keep only finite spectra
            valid_mask = np.all(np.isfinite(class_augmented), axis=1)
            class_augmented = class_augmented[valid_mask]

            if len(class_augmented) == 0:
                continue

            # keep preprocessing consistent
            class_augmented = normalize(class_augmented, norm='max')
            original_norm = normalize(class_spectra, norm='max')

            # save plot example from first class only
            if itr == 0:
                class_original_for_plot = original_norm.copy()
                class_emsc_for_plot = emsc_spectra.copy()
                class_aug_for_plot = class_augmented.copy()

            y_add = np.full(len(class_augmented), itr)

            augmented_x_list.append(class_augmented)
            augmented_y_list.append(y_add)

        # normalize original train/test too, to keep consistency
        # x_train_norm = normalize(x_train, norm='max')
        # x_test_norm = normalize(x_test, norm='max')

        if len(augmented_x_list) > 0:
            x_aug = np.concatenate(augmented_x_list, axis=0)
            y_aug = np.concatenate(augmented_y_list, axis=0)

            x_train_final = np.concatenate((x_train, x_aug), axis=0)
            y_train_final = np.concatenate((y_train, y_aug), axis=0)
        else:
            x_train_final = x_train
            y_train_final = y_train

        model = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
        model.fit(x_train_final, y_train_final)
        y_pre = model.predict(x_test)

        scores = utils.printScore(y_test, y_pre)
        cm = confusion_matrix(y_test, y_pre)

        cmTotal += cm
        scoreTotal += scores
        m += 1
        print(f'Finished seed {seedNum}, run count = {m}')

    print('Average scores:', scoreTotal / m)

    cmTotal = cmTotal / m
    utils.plot_confusion_matrix(cmTotal, PN, 'PCA-EMSA data_SVM')

    # plot examples
    if class_original_for_plot is not None and class_emsc_for_plot is not None and class_aug_for_plot is not None:
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

        for item in class_emsc_for_plot[:10]:
            ax[0].plot(waveLength, item)
        for item in class_original_for_plot[:10]:
            ax[1].plot(waveLength, item)
        for item in class_aug_for_plot[:10]:
            ax[2].plot(waveLength, item)

        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[2].tick_params(labelsize=12)

        ax[0].set_title('EMSC spectra', font2)
        ax[1].set_title('Original spectra', font2)
        ax[2].set_title('PCA-EMSA augmented spectra', font2)

        plt.tight_layout()
        plt.show()