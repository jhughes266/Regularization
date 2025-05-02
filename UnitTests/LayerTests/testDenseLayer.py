import unittest
import numpy as np
import numpy.testing as np_testing
from Layers.DenseLayer import DenseLayer
class DenseLayerTest(unittest.TestCase):
    def setUp(self):
        self.denseLayerTest = DenseLayer(3, 2, "Sigmoid", 7, 0.3)
        self.denseLayerTest._w = np.array([[0.1, -0.3, 0.6], [0.7, -0.4, -0.5]])
        self.denseLayerTest._b = np.array([[1.2], [0.4]])

    def testForward(self):
        x = np.array([[0.5], [0.9], [-0.8]])
        desiredY = np.array([[0.622459331], [0.687831331]])
        actualY = self.denseLayerTest.Forward(x)
        np.testing.assert_almost_equal(actualY, desiredY)

    def testBackward(self):
        dy = np.array([[-0.6], [0.55]])
        x = np.array([[0.5], [0.9], [-0.8]])
        self.denseLayerTest.Forward(x)
        actualDx = self.denseLayerTest.Backward(dy)
        actualDw = self.denseLayerTest._dw
        actualDb = self.denseLayerTest._db

        desiredDx = np.array([[0.068566743],[-0.004937598],[-0.143649169]])
        desriedDw =np.array([[-0.070501114, -0.126902005, 0.112801782],[0.059047833, 0.106286099, -0.094476532]])
        desiredDb = np.array([[-0.141002227],[0.118095665]])


        np_testing.assert_almost_equal(actualDx, desiredDx)
        np_testing.assert_almost_equal(actualDw, desriedDw)
        np_testing.assert_almost_equal(actualDb, desiredDb)

    def testClip(self):
        clipThreshold = 0.4
        self.denseLayerTest._dw = np.array([[0.1, -0.5, 0.3], [-1.3, 1.2, 0.6]])
        self.denseLayerTest._db = np.array([[1.6], [-0.2]])

        self.denseLayerTest.GradientClipping("Clip", clipThreshold)

        desiredClippedDw = np.array([[0.1, -0.4, 0.3], [-0.4, 0.4, 0.4]])
        desiredClippedDb = np.array([[0.4], [-0.2]])

        actualClippedDw = self.denseLayerTest._dw
        actualClippedDb = self.denseLayerTest._db

        np_testing.assert_almost_equal(actualClippedDw, desiredClippedDw)
        np_testing.assert_almost_equal(actualClippedDb, desiredClippedDb)

    def testNorm(self):
        clipThreshold = 0.4
        self.denseLayerTest._dw = np.array([[0.1, -0.5, 0.3], [-1.3, 1.2, 0.6]])
        self.denseLayerTest._db = np.array([[1.6], [-0.2]])

        self.denseLayerTest.GradientClipping("Norm", clipThreshold)

        actualGradientNorm = self.denseLayerTest._gradientNorm
        desiredGradientNorm = 2.537715508

        desiredSacledDw = np.array([[0.015762208, -0.078811041, 0.047286624], [-0.204908706, 0.189146497, 0.094573249]])
        desiredSacledDb = np.array([[0.25219533], [-0.031524416]])

        actualSacledDw = self.denseLayerTest._dw
        actualSacledDb = self.denseLayerTest._db

        np_testing.assert_almost_equal(actualGradientNorm, desiredGradientNorm)
        np_testing.assert_almost_equal(desiredSacledDb, actualSacledDb)
        np_testing.assert_almost_equal(desiredSacledDw, actualSacledDw)

    def testGradientStepL1(self):
        lam = 0.1
        self.denseLayerTest._dw = np.array([[0.1, -0.5, 0.3], [-1.3, 1.2, 0.6]])
        self.denseLayerTest._db = np.array([[1.6], [-0.2]])

        self.denseLayerTest.GradientStep(L1=True, lam=0.1)
        desiredW = np.array([0.065714286, -0.248571429, 0.557142857, 0.725714286, -0.421428571, -0.495714286]).reshape(2, 3)
        desiredB = np.array([[1.131428571], [0.408571429]])

        actualW = self.denseLayerTest._w
        actualB = self.denseLayerTest._b

        np_testing.assert_almost_equal(desiredW, actualW)
        np_testing.assert_almost_equal(desiredB, actualB)

    def testGradientStepL2(self):
        lam = 0.1
        self.denseLayerTest._dw = np.array([[0.1, -0.5, 0.3], [-1.3, 1.2, 0.6]])
        self.denseLayerTest._db = np.array([[1.6], [-0.2]])

        self.denseLayerTest.GradientStep(L2=True, lam=0.1)
        desiredW = np.array([0.089714286, -0.260571429, 0.551142857, 0.713714286, -0.427428571, -0.495714286]).reshape(2, 3)
        desiredB = np.array([[1.131428571], [0.408571429]])

        actualW = self.denseLayerTest._w
        actualB = self.denseLayerTest._b

        np_testing.assert_almost_equal(desiredW, actualW)
        np_testing.assert_almost_equal(desiredB, actualB)


    def testReset(self):
        self.denseLayerTest._dw = np.array([[0.1, -0.5, 0.3], [-1.3, 1.2, 0.6]])
        self.denseLayerTest._db = np.array([[1.6], [-0.2]])
        self.denseLayerTest.Reset()

        desiredDw = np.zeros((2, 3))
        desiredDb = np.zeros((2, 1))

        actualDw = self.denseLayerTest._dw
        actualDb = self.denseLayerTest._db

        np_testing.assert_almost_equal(desiredDw, actualDw)
        np_testing.assert_almost_equal(desiredDb, actualDb)

    def testInvlaidClipType(self):
        clipThreshold = 0.4
        self.denseLayerTest._dw = np.array([[0.1, -0.5, 0.3], [-1.3, 1.2, 0.6]])
        self.denseLayerTest._db = np.array([[1.6], [-0.2]])

        with self.assertRaises(ValueError):
            self.denseLayerTest.GradientClipping("wrong", clipThreshold)

    def testMultipleForwardBackward(self):
        #setup
        x = np.array([
            [[0.1], [0.54], [0.73]],
            [[0.45], [0.34], [0.89]],
            [[0.683], [0.47], [0.83]]
        ])
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        dy = np.array([
            [[0.234], [0.786]],
            [[-0.3423], [-0.987]],
            [[-0.12], [0.45]]
        ])
        dy1 = dy[0]
        dy2 = dy[1]
        dy3 = dy[2]

        actualY1 = self.denseLayerTest.Forward(x1)
        desiredY1 = np.array([0.815477135, 0.472278457]).reshape(2, 1)
        actualDX1 = self.denseLayerTest.Backward(dy1)
        desiredDX1 = np.array([0.140648276, -0.088921676, -0.076821411]).reshape(3, 1)
        np_testing.assert_almost_equal(actualY1, desiredY1)
        np_testing.assert_almost_equal(actualDX1, desiredDX1)

        actualY2 = self.denseLayerTest.Forward(x2)
        desiredY2 = np.array([0.842506873, 0.533449963]).reshape(2, 1)
        actualDX2 = self.denseLayerTest.Backward(dy2)
        desiredDX2 = np.array([-0.176493898, 0.111884096, 0.095571147]).reshape(3, 1)
        np_testing.assert_almost_equal(actualY2, desiredY2)
        np_testing.assert_almost_equal(actualDX2, desiredDX2)

        actualY3 = self.denseLayerTest.Forward(x3)
        desiredY3 = np.array([0.835524768, 0.568344517]).reshape(2, 1)
        actualDX3 = self.denseLayerTest.Backward(dy3)
        desiredDX3 = np.array([0.075629566, -0.039211992, -0.065093496]).reshape(3, 1)
        np_testing.assert_almost_equal(actualY3, desiredY3)
        np_testing.assert_almost_equal(actualDX3, desiredDX3)

        desiredDb = np.array([-0.026699277, 0.060648388]).reshape(2, 1)
        desiredDw = np.array([-0.028180861, -0.004179364, -0.028406663, -0.015549067, 0.074151394, 0.016009826]).reshape(2, 3)
        actualDb = self.denseLayerTest._db
        actualDw = self.denseLayerTest._dw

        np_testing.assert_almost_equal(actualDw, desiredDw)
        np_testing.assert_almost_equal(actualDb, desiredDb)













