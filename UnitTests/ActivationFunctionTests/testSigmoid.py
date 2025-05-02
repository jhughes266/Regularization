import unittest
import numpy as np
import numpy.testing as np_testing
from ActivationFunctions.Sigmoid import Sigmoid
class SigmoidTest(unittest.TestCase):
    def setUp(self):
        self.sigmoidTest = Sigmoid()

    def testForward(self):
        x = np.array([-.5, .25, .7]).reshape((3, 1))
        desiredY = np.array([0.377540669, 0.562176501, 0.668187772]).reshape((3, 1))
        actualY = self.sigmoidTest.Forward(x)
        np_testing.assert_almost_equal(actualY, desiredY, decimal = 6)

    def testBackward(self):
        x = np.array([-.5, .25, .7]).reshape((3, 1))
        self.sigmoidTest.Forward(x)

        dy = np.array([0.5, -2.3, 1.7]).reshape((3, 1))
        desiredDX = np.array([0.117501856, -0.56610839, 0.376911885]).reshape((3, 1))
        actualDX = self.sigmoidTest.Backward(dy)
        np_testing.assert_almost_equal(actualDX, desiredDX)

    def testMultipleForwardAndBackward(self):
        #EXAMPLE 1
        #forward
        x1 = np.array([-.5, .25, .7]).reshape((3, 1))
        desiredY1 = np.array([0.377540669, 0.562176501, 0.668187772]).reshape((3, 1))
        actualY1 = self.sigmoidTest.Forward(x1)
        np_testing.assert_almost_equal(actualY1, desiredY1, decimal=6)
        #backward
        dy1 = np.array([-.5, .25, .7]).reshape((3, 1))
        desiredDX1 = np.array([-0.117501856, 0.061533521, 0.155199011]).reshape((3, 1))
        actualDX1 = self.sigmoidTest.Backward(dy1)
        np_testing.assert_almost_equal(actualDX1, desiredDX1, decimal=6)

        #EXAMPLE 2
        #forward
        x2 = np.array([-.6, .7, .4]).reshape((3, 1))
        desiredY2 = np.array([0.354343694, 0.668187772, 0.59868766]).reshape((3, 1))
        actualY2 = self.sigmoidTest.Forward(x2)
        np_testing.assert_almost_equal(actualY2, desiredY2, decimal=6)
        #backward
        dy2 = np.array([-.6, .7, .4]).reshape((3, 1))
        desiredDX2 = np.array([-0.137270544, 0.155199011, 0.096104298]).reshape((3, 1))
        actualDX2 = self.sigmoidTest.Backward(dy2)
        np_testing.assert_almost_equal(actualDX2, desiredDX2, decimal=6)

        #EXAMPLE 3
        # forward
        x3 = np.array([-.7, -.5, -.7]).reshape((3, 1))
        desiredY3 = np.array([0.331812228, 0.377540669, 0.331812228]).reshape((3, 1))
        actualY3 = self.sigmoidTest.Forward(x3)
        np_testing.assert_almost_equal(actualY3, desiredY3, decimal=6)
        # backward
        dy3 = np.array([-.7, -.5, -.7]).reshape((3, 1))
        desiredDX3 = np.array([-0.155199011, -0.117501856, -0.155199011]).reshape((3, 1))
        actualDX3 = self.sigmoidTest.Backward(dy3)
        np_testing.assert_almost_equal(actualDX3, desiredDX3, decimal=6)

