import unittest
import numpy as np
import numpy.testing as np_testing
from Layers.CostLayer import MSECostLayer

class CostLayerTest(unittest.TestCase):
    def setUp(self):
        self.costLayerTest = MSECostLayer(3)

    def testForward(self):
        y = np.array([1, 0, 1]).reshape(3, 1)
        yhat = np.array([0.648713633, 0.56211919, 0.699153748]).reshape(3, 1)
        self.costLayerTest.Forward(y, yhat)
        actualCost = self.costLayerTest._costSum
        desiredCost = 0.264944281
        np_testing.assert_almost_equal(actualCost, desiredCost)

    def testBackward(self):
        y = np.array([1, 0, 1]).reshape(3, 1)
        yhat = np.array([0.648713633, 0.56211919, 0.699153748]).reshape(3, 1)
        self.costLayerTest.Forward(y, yhat)
        actualDyhat = self.costLayerTest.Backward()
        desiredDyhat = np.array([-0.351286367, 0.56211919, -0.300846252]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat, desiredDyhat)

    def testMultipleForwardBackward(self):
        #1
        y1 = np.array([1, 0, 1]).reshape(3, 1)
        yhat1 = np.array([0.648713633, 0.56211919, 0.699153748]).reshape(3, 1)
        self.costLayerTest.Forward(y1, yhat1)
        actualDyhat1 = self.costLayerTest.Backward()
        desiredDyhat1 = np.array([-0.351286367, 0.56211919, -0.300846252]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat1, desiredDyhat1)
        #2
        y2 = np.array([1, 0, 0]).reshape(3, 1)
        yhat2 = np.array([0.642287211, 0.556126047, 0.700581329]).reshape(3, 1)
        self.costLayerTest.Forward(y2, yhat2)
        actualDyhat2 = self.costLayerTest.Backward()
        desiredDyhat2 = np.array([-0.357712789, 0.556126047, 0.700581329]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat2, desiredDyhat2)
        #3
        y3 = np.array([0, 0, 1]).reshape(3, 1)
        yhat3 =  np.array([0.646291875, 0.559777903, 0.699818357]).reshape(3, 1)
        self.costLayerTest.Forward(y3, yhat3)
        actualDyhat3 = self.costLayerTest.Backward()
        desiredDyhat3 = np.array([0.646291875, 0.559777903, -0.300181643]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat3, desiredDyhat3)

        actualTotalSumCost = self.costLayerTest.CostSum
        desiredTotalSumCost = 1.139545443
        np_testing.assert_almost_equal(actualTotalSumCost, desiredTotalSumCost)

    def testMultipleForwardBackwardWithL1(self):
        #1
        y1 = np.array([1, 0, 1]).reshape(3, 1)
        yhat1 = np.array([0.648713633, 0.56211919, 0.699153748]).reshape(3, 1)
        self.costLayerTest.Forward(y1, yhat1)
        actualDyhat1 = self.costLayerTest.Backward()
        desiredDyhat1 = np.array([-0.351286367, 0.56211919, -0.300846252]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat1, desiredDyhat1)
        #2
        y2 = np.array([1, 0, 0]).reshape(3, 1)
        yhat2 = np.array([0.642287211, 0.556126047, 0.700581329]).reshape(3, 1)
        self.costLayerTest.Forward(y2, yhat2)
        actualDyhat2 = self.costLayerTest.Backward()
        desiredDyhat2 = np.array([-0.357712789, 0.556126047, 0.700581329]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat2, desiredDyhat2)
        #3
        y3 = np.array([0, 0, 1]).reshape(3, 1)
        yhat3 =  np.array([0.646291875, 0.559777903, 0.699818357]).reshape(3, 1)
        self.costLayerTest.Forward(y3, yhat3)
        actualDyhat3 = self.costLayerTest.Backward()
        desiredDyhat3 = np.array([0.646291875, 0.559777903, -0.300181643]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat3, desiredDyhat3)

        actualTotalSumCost = self.costLayerTest.CostSum
        desiredTotalSumCost = 1.139545443
        np_testing.assert_almost_equal(actualTotalSumCost, desiredTotalSumCost)

        w1 = np.array([0.1, -0.67, 0.02, 0.9, -0.45, -0.34]).reshape(3, 2)
        w2 = np.array([0.23, 0.12, 0.56, -0.65, -0.8, 0.8]).reshape(2, 3)
        w3 = np.array([0.4, 0.5, -0.23, 0.67, 0.97, -0.56]).reshape(3, 2)
        modelWeights = [w1, w2, w3]

        actualL1Cost, _ = self.costLayerTest.DisplayInfo(display=True, modelWeights=modelWeights, lam=0.1, L1=True)
        desiredL1Cost = 1.276848481
        np_testing.assert_almost_equal(actualL1Cost, desiredL1Cost)

    def testMultipleForwardBackwardWithL2(self):
        # 1
        y1 = np.array([1, 0, 1]).reshape(3, 1)
        yhat1 = np.array([0.648713633, 0.56211919, 0.699153748]).reshape(3, 1)
        self.costLayerTest.Forward(y1, yhat1)
        actualDyhat1 = self.costLayerTest.Backward()
        desiredDyhat1 = np.array([-0.351286367, 0.56211919, -0.300846252]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat1, desiredDyhat1)
        # 2
        y2 = np.array([1, 0, 0]).reshape(3, 1)
        yhat2 = np.array([0.642287211, 0.556126047, 0.700581329]).reshape(3, 1)
        self.costLayerTest.Forward(y2, yhat2)
        actualDyhat2 = self.costLayerTest.Backward()
        desiredDyhat2 = np.array([-0.357712789, 0.556126047, 0.700581329]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat2, desiredDyhat2)
        # 3
        y3 = np.array([0, 0, 1]).reshape(3, 1)
        yhat3 = np.array([0.646291875, 0.559777903, 0.699818357]).reshape(3, 1)
        self.costLayerTest.Forward(y3, yhat3)
        actualDyhat3 = self.costLayerTest.Backward()
        desiredDyhat3 = np.array([0.646291875, 0.559777903, -0.300181643]).reshape(3, 1)
        np_testing.assert_almost_equal(actualDyhat3, desiredDyhat3)

        actualTotalSumCost = self.costLayerTest.CostSum
        desiredTotalSumCost = 1.139545443
        np_testing.assert_almost_equal(actualTotalSumCost, desiredTotalSumCost)

        w1 = np.array([0.1, -0.67, 0.02, 0.9, -0.45, -0.34]).reshape(3, 2)
        w2 = np.array([0.23, 0.12, 0.56, -0.65, -0.8, 0.8]).reshape(2, 3)
        w3 = np.array([0.4, 0.5, -0.23, 0.67, 0.97, -0.56]).reshape(3, 2)
        modelWeights = [w1, w2, w3]

        actualL2Cost, _ = self.costLayerTest.DisplayInfo(display=True, modelWeights=modelWeights, lam=0.1, L2=True)
        desiredL2Cost = 0.963558481
        np_testing.assert_almost_equal(actualL2Cost, desiredL2Cost)





