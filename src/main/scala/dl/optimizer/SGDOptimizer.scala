package dl.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}

class SGDOptimizer extends Optimizer {

  override def update(weightMat: DenseMatrix[Double], dWeightMat: DenseMatrix[Double],learnRate:Double): DenseMatrix[Double] = {
    weightMat - (DenseMatrix.tabulate(dWeightMat.rows,dWeightMat.cols)((x,y) => learnRate) *:* dWeightMat)
  }

  override def update(offsetVec: DenseVector[Double], dOffsetVec: DenseVector[Double],learnRate:Double): DenseVector[Double] = {
    offsetVec - (learnRate * dOffsetVec)
  }

}
