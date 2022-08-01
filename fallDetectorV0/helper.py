import numpy as np
import cv2 as cv
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV



class MHI_Generator:
    '''
    Create the MHI for the fall detector
    '''
    def __init__(self, dim=128, threshold=0.1, interval=2, win_size=40):
        # initialize MHI params
        self.index = 0
        self.dim = dim
        self.threshold = threshold
        self.interval = interval
        self.win_size = win_size
        self.decay = 1.0/self.win_size

        # Initialize mhi frame
        self.mhi_zeros = np.zeros((dim, dim))

    def process(self, frame, save_batch=True):
        self.index += 1
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if self.index == 1:
            self.prev_frame = cv.resize(frame,(self.dim, self.dim),interpolation=cv.INTER_AREA)
            self.prev_mhi = self.mhi_zeros

        if self.index % self.interval == 0:
            frame = cv.resize(frame,(self.dim, self.dim),
                                         interpolation=cv.INTER_AREA)
            diff = cv.absdiff(self.prev_frame, frame)
            binary = (diff >= (self.threshold * 255)).astype(np.uint8)
            mhi = binary + (binary == 0) * np.maximum(self.mhi_zeros,(self.prev_mhi - self.decay))

            # Update mhi frame
            self.prev_frame = frame
            self.prev_mhi = mhi

            if self.index >= (self.win_size * self.interval):
                img = cv.normalize(mhi, None, 0.0, 255.0, cv.NORM_MINMAX)
                if save_batch:
                    return img
                else:
                    flag, encode_img = cv.imencode('.png', img)
                    if flag:
                        return bytearray(encode_img)
        return None


class ContourFeatureExtractor:
    '''
    Process MHI frames to create contour features anf return a feature vector (Num_frames, Num_features)
    '''
    def __init__(self, debug=False):
        self.debug = debug
        self.pcaMean = None
        self.pcaEig = None

    def getFrame(self, frame_path):
        frame = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        if frame is None:
            print(f'Error: Image path invalid: {frame_path}')

        return frame

    def process(self, frame):

        # Binary thresh
        _, output_frame = cv.threshold(frame, 200, 255, cv.THRESH_BINARY)
        output_frame = output_frame.astype(np.uint8)

        # Get the number of non-zero pixels
        nonZeroCount = cv.countNonZero(output_frame)

        if self.debug:
            cv.imshow(f'Thresh', output_frame)

        # Morphology
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        output_frame = cv.morphologyEx(output_frame, cv.MORPH_CLOSE, kernel, iterations=2)
        output_frame = cv.dilate(output_frame, kernel, iterations=1)

        if self.debug:
            cv.imshow(f'Morph', output_frame)

        # Get contours
        conts, _ = cv.findContours(output_frame, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        conts = sorted(conts, key = cv.contourArea, reverse=True)

        if len(conts) is not 0:
            area = cv.contourArea(conts[0])
            M = cv.moments(conts[0])

            # Draw rectangle
            (x, y, w, h) = cv.boundingRect(conts[0])
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            if self.debug:
                cv.imshow(f'Bounding box, Area: {area}', frame)
                cv.waitKey(0)
                cv.destroyAllWindows()

            # feature array: area, nonZeroCount, Perimeter, Width, Ratio W/H, 24 Moment values
            return [area, nonZeroCount, cv.arcLength(conts[0],True), w, float(w/h)] + list(M.values())
        else:
            return None

    def dimensionReduction(self, features):
        # To handle data reduction for 1 image at a time
        features = np.array(features)
        if len(features.shape) == 1:
            return cv.PCAProject(np.array(features).T, self.pcaMean.T, self.pcaEig)
        else:
            return cv.PCAProject(features, self.pcaMean, self.pcaEig)

    def computePCA(self, features, numFeaturesToKeep=2):
        mean = np.empty((0))
        if numFeaturesToKeep is None:
            self.pcaMean, self.pcaEig, _ = cv.PCACompute2(np.array(features), mean, retainedVariance=0.99)
        else:
            self.pcaMean, self.pcaEig, _ = cv.PCACompute2(np.array(features), mean, maxComponents=numFeaturesToKeep)

    def savePCAToFile(self, path):
        if self.pcaMean is not None and self.pcaEig is not None:
            dump(self.pcaMean, f'{path}/pcaMean.joblib')
            dump(self.pcaEig, f'{path}/pcaEig.joblib')

    def loadPCAFromFile(self, filepath):
        self.pcaMean = load(f'{filepath}/pcaMean.joblib')
        self.pcaEig = load(f'{filepath}/pcaEig.joblib')

    def calculateStats(self, features, printResult=False):
        m = np.mean(features, axis=0)
        v = np.var(features, axis=0)

        if printResult:
            print(m)
            print(v)

        return (m, v)

    def plot2dScatter(self, fallFeat, notFallFeat):
        fig, axs = plt.subplots(nrows=3, ncols=1)
        fig.suptitle('scatter 2d Data')

        axs[0].scatter((fallFeat[:,0]), (fallFeat[:,1]), color='green', alpha=0.3, label='fall')
        axs[1].scatter((notFallFeat[:,0]), (notFallFeat[:,1]), color='blue', alpha=0.3, label='not fall')

        axs[2].scatter((fallFeat[:,0]), (fallFeat[:,1]), color='green', alpha=0.3, label='fall')
        axs[2].scatter((notFallFeat[:,0]), (notFallFeat[:,1]), color='blue', alpha=0.3, label='not fall')
        plt.legend()
        plt.show()


class SVM_classifier:
    def __init__(self, kernel='rbf', gamma='scale', C=0.8):
        self.clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, class_weight='balanced')

    def train(self, data, labels):
        self.clf.fit(data, labels)

    def crossFoldValidation(self, data, labels, cv=3):
        scores = cross_val_score(self.clf, data, labels, cv=cv)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    def gridSearch(self, data, labels):
        parameters =  {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]}
        gs = GridSearchCV(self.clf, parameters)
        gs.fit(data, labels)
        print(gs.cv_results_)

    def predict(self, data):
        if data.shape[0] != 1 and data.shape[1] == 1:
            data = data.T
        return self.clf.predict(data)

    def results(self, actual, predicted, data, name=''):
        cmatrix = metrics.confusion_matrix(actual, predicted, normalize='true')

        # Setting the attributes
        _, px = plt.subplots(figsize=(7.5, 7.5))
        px.matshow(cmatrix, cmap=plt.cm.YlOrRd, alpha=0.5)
        for m in range(cmatrix.shape[0]):
            for n in range(cmatrix.shape[1]):
                px.text(x=m,y=n,s="{:.3f}".format(cmatrix[m, n]), va='center', ha='center', size='xx-large')

        # Sets the labels
        plt.xlabel('Predictions', fontsize=16)
        plt.ylabel('Actuals', fontsize=16)
        plt.title(f'{name} Confusion Matrix', fontsize=15)
        plt.show()

        print(self.clf.score(data, actual))

        return cmatrix

    def saveSVMToFile(self, path=''):
        dump(self.clf, f'{path}/svm.joblib')

    def loadSvmFromFile(self, filepath):
        self.clf = load(f'{filepath}/svm.joblib')

    '''
    Helper function for visualization
    '''
    def plot_contours(self, data, labels):
        """Plot the decision boundaries for a classifier."""
        print("visualizing")
        data = preprocessing.normalize(data, norm='max', axis=0)
        X0, X1 = data[:, 0], data[:, 1]
        xx, yy = make_meshgrid(X0, X1, h=0.1)

        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            aspect="auto",
            origin="lower",
            cmap=plt.cm.PuOr_r,
        )
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
        plt.scatter(X0, X1, s=30, c=labels, cmap=plt.cm.Paired, edgecolors="k")
        plt.show()

    def plotDecisionBoundary(self, X, y):
        # plot the samples
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

        # plot the decision functions for both classifiers
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        # get the separating hyperplane for weighted classes
        #Z = self.clf.decision_function(xy).reshape(XX.shape)

        if hasattr(self.clf, "decision_function"):
            Z = self.clf.decision_function(xy).reshape(XX.shape)
        else:
            Z = self.clf.predict_proba(xy)[:, 1].reshape(XX.shape)

        # plot decision boundary and margins for weighted classes
        ax.contourf(XX, YY, Z, cmap=plt.cm.RdBu, alpha=0.8)
        ax.contour(XX, YY, Z, colors="r", levels=[0], alpha=0.5, linestyles=["-"])

        plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy