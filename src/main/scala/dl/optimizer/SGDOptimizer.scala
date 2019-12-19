package dl.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}

class SGDOptimizer(val learnRate:Double) extends Optimizer {

  override def update(layerIdx:Int,weightMat: DenseMatrix[Double], dWeightMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    weightMat - (DenseMatrix.tabulate(dWeightMat.rows,dWeightMat.cols)((x,y) => learnRate) *:* dWeightMat)
  }

  override def update(layerIdx:Int,offsetVec: DenseVector[Double], dOffsetVec: DenseVector[Double]): DenseVector[Double] = {
    offsetVec - (learnRate * dOffsetVec)
  }

}
