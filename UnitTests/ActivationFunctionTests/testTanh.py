import unittest
import numpy as np
import numpy.testing as np_testing
from ActivationFunctions.Tanh import Tanh
class SigmoidTest(unittest.TestCase):
    def setUp(self):
        self.tanhTest = Tanh()

    def testForward(self):
        x = np.array([-.5, .25, .7]).reshape((3, 1))
        desiredY = np.array([-0.462117157, 0.244918662, 0.604367777]).reshape((3, 1))
        actualY = self.tanhTest.Forward(x)
        np_testing.assert_almost_equal(actualY, desiredY, decimal=6)

    def testBackward(self):
        x = np.array([-.5, .25, .7]).reshape((3, 1))
        self.tanhTest.Forward(x)

        dy = np.array([0.5, -2.3, 1.7]).reshape((3, 1))
        desiredDX = np.array([0.393223866, -2.162034152, 1.079057303]).reshape((3, 1))
        actualDX = self.tanhTest.Backward(dy)
        np_testing.assert_almost_equal(actualDX, desiredDX)

    def testMultipleForwardAndBackward(self):
        #EXAMPLE 1
        #forward
        x1 = np.array([-.5, .25, .7]).reshape((3, 1))
        desiredY1 = np.array([-0.462117157, 0.244918662, 0.604367777]).reshape((3, 1))
        actualY1 = self.tanhTest.Forward(x1)
        np_testing.assert_almost_equal(actualY1, desiredY1, decimal=6)
        #backward
        dy1 = np.array([-.5, .25, .7]).reshape((3, 1))
        desiredDX1 = np.array([-0.393223866, 0.235003712, 0.444317713]).reshape((3, 1))
        actualDX1 = self.tanhTest.Backward(dy1)
        np_testing.assert_almost_equal(actualDX1, desiredDX1, decimal=6)

        #EXAMPLE 2
        #forward
        x2 = np.array([-.6, .7, .4]).reshape((3, 1))
        desiredY2 = np.array([-0.537049567, 0.604367777, 0.379948962]).reshape((3, 1))
        actualY2 = self.tanhTest.Forward(x2)
        np_testing.assert_almost_equal(actualY2, desiredY2, decimal=6)
        #backward
        dy2 = np.array([-.6, .7, .4]).reshape((3, 1))
        desiredDX2 = np.array([-0.426946658, 0.444317713, 0.342255514]).reshape((3, 1))
        actualDX2 = self.tanhTest.Backward(dy2)
        np_testing.assert_almost_equal(actualDX2, desiredDX2, decimal=6)

        #EXAMPLE 3
        # forward
        x3 = np.array([-.7, -.5, -.7]).reshape((3, 1))
        desiredY3 = np.array([-0.604367777, -0.462117157, -0.604367777]).reshape((3, 1))
        actualY3 = self.tanhTest.Forward(x3)
        np_testing.assert_almost_equal(actualY3, desiredY3, decimal=6)
        # backward
        dy3 = np.array([-.7, -.5, -.7]).reshape((3, 1))
        desiredDX3 = np.array([-0.444317713, -0.393223866, -0.444317713]).reshape((3, 1))
        actualDX3 = self.tanhTest.Backward(dy3)
        np_testing.assert_almost_equal(actualDX3, desiredDX3, decimal=6)

